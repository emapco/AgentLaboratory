import os
import pickle
import sys
import time
from copy import copy

import hydra
from omegaconf import DictConfig
from torch.backends.mkl import verbose

from agents import (
    MLEngineerAgent,
    PhDStudentAgent,
    PostdocAgent,
    ProfessorAgent,
    ReviewersAgent,
    SWEngineerAgent,
)
from common_imports import *  # noqa: F403
from mlesolver import MLESolver
from papersolver import PaperSolver
from tools import ArxivSearch, HFDataSearch, execute_code
from utils import extract_prompt, remove_directory, remove_figures, save_to_file


def setup_logging():
    class PrintLogger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "a")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = PrintLogger("agent_lab_log.txt")


class LaboratoryWorkflow:
    def __init__(
        self,
        research_topic,
        openai_api_key,
        default_llm_backend: str,
        agent_model_backbone: dict[str, str],
        max_steps=100,
        num_papers_lit_review=5,
        notes=list(),
        human_in_loop_flag: dict | None = None,
        compile_pdf=True,
        verbose=True,
        print_costs=True,
        mlesolver_max_steps=3,
        papersolver_max_steps=5,
    ):
        """
        Initialize laboratory workflow
        @param research_topic: (str) description of research idea to explore
        @param max_steps: (int) max number of steps for each phase, i.e. compute tolerance budget
        @param num_papers_lit_review: (int) number of papers to include in the lit review
        @param agent_model_backbone: (str or dict) model backbone to use for agents
        @param notes: (list) notes for agent to follow during tasks
        """

        self.notes = notes
        self.max_steps = max_steps
        self.compile_pdf = compile_pdf
        self.openai_api_key = openai_api_key
        self.research_topic = research_topic
        self.default_llm_backend = default_llm_backend
        self.model_backbone = agent_model_backbone
        self.num_papers_lit_review = num_papers_lit_review

        self.print_cost = print_costs
        self.review_override = True  # should review be overridden?
        self.review_ovrd_steps = 0  # review steps so far
        self.arxiv_paper_exp_time = 3
        self.reference_papers = list()

        ##########################################
        ####### COMPUTE BUDGET PARAMETERS ########
        ##########################################
        self.num_ref_papers = 1
        self.review_total_steps = 0  # num steps to take if overridden
        self.arxiv_num_summaries = 5
        self.mlesolver_max_steps = mlesolver_max_steps
        self.papersolver_max_steps = papersolver_max_steps

        self.phases = [
            ("literature review", ["literature review"]),
            ("plan formulation", ["plan formulation"]),
            ("experimentation", ["data preparation", "running experiments"]),
            (
                "results interpretation",
                ["results interpretation", "report writing", "report refinement"],
            ),
        ]
        self.phase_status = dict()
        for phase, subtasks in self.phases:
            for subtask in subtasks:
                self.phase_status[subtask] = False

        # validate agent_model_backbone contains a llm for each subtask
        valid_subtasks = [subtask for phase in self.phases for subtask in phase[1]]
        for phase, llm in agent_model_backbone.items():
            if phase not in valid_subtasks:
                raise ValueError(f"Invalid phase: {phase}")
            valid_subtasks.remove(phase)
        if len(valid_subtasks) != 0:
            raise ValueError(f"Missing subtasks: {valid_subtasks}")

        self.phase_models = agent_model_backbone
        self.human_in_loop_flag: dict = human_in_loop_flag or dict()

        self.statistics_per_phase = {
            "literature review": {
                "time": 0.0,
                "steps": 0.0,
            },
            "plan formulation": {
                "time": 0.0,
                "steps": 0.0,
            },
            "data preparation": {
                "time": 0.0,
                "steps": 0.0,
            },
            "running experiments": {
                "time": 0.0,
                "steps": 0.0,
            },
            "results interpretation": {
                "time": 0.0,
                "steps": 0.0,
            },
            "report writing": {
                "time": 0.0,
                "steps": 0.0,
            },
            "report refinement": {
                "time": 0.0,
                "steps": 0.0,
            },
        }

        self.save = True
        self.verbose = verbose
        self.reviewers = ReviewersAgent(
            model=self.model_backbone,  # type: ignore
            notes=self.notes,
            openai_api_key=self.openai_api_key,
        )
        self.phd = PhDStudentAgent(
            model=self.model_backbone,  # type: ignore
            notes=self.notes,
            max_steps=self.max_steps,
            openai_api_key=self.openai_api_key,
        )
        self.postdoc = PostdocAgent(
            model=self.model_backbone,  # type: ignore
            notes=self.notes,
            max_steps=self.max_steps,
            openai_api_key=self.openai_api_key,
        )
        self.professor = ProfessorAgent(
            model=self.model_backbone,  # type: ignore
            notes=self.notes,
            max_steps=self.max_steps,
            openai_api_key=self.openai_api_key,
        )
        self.ml_engineer = MLEngineerAgent(
            model=self.model_backbone,  # type: ignore
            notes=self.notes,
            max_steps=self.max_steps,
            openai_api_key=self.openai_api_key,
        )
        self.sw_engineer = SWEngineerAgent(
            model=self.model_backbone,  # type: ignore
            notes=self.notes,
            max_steps=self.max_steps,
            openai_api_key=self.openai_api_key,
        )

        # remove previous files
        remove_figures()
        remove_directory("research_dir")
        # make src and research directory
        if not os.path.exists("state_saves"):
            os.mkdir("state_saves")
        os.mkdir(os.path.join("research_dir"))
        os.mkdir(os.path.join("research_dir", "src"))
        os.mkdir(os.path.join("research_dir", "tex"))

    def set_model(self, model):
        self.set_agent_attr("model", model)
        self.reviewers.model = model

    def save_state(self, phase):
        """
        Save state for phase
        @param phase: (str) phase string
        @return: None
        """
        phase = phase.replace(" ", "_")
        with open(f"state_saves/{phase}.pkl", "wb") as f:
            pickle.dump(self, f)

    def set_agent_attr(self, attr, obj):
        """
        Set attribute for all agents
        @param attr: (str) agent attribute
        @param obj: (object) object attribute
        @return: None
        """
        setattr(self.phd, attr, obj)
        setattr(self.postdoc, attr, obj)
        setattr(self.professor, attr, obj)
        setattr(self.ml_engineer, attr, obj)
        setattr(self.sw_engineer, attr, obj)

    def reset_agents(self):
        """
        Reset all agent states
        @return: None
        """
        self.phd.reset()
        self.postdoc.reset()
        self.professor.reset()
        self.ml_engineer.reset()
        self.sw_engineer.reset()

    def perform_research(self):
        """
        Loop through all research phases
        @return: None
        """
        for phase, subtasks in self.phases:
            phase_start_time = time.time()  # Start timing the phase
            if self.verbose:
                print(f"{'*' * 50}\nBeginning phase: {phase}\n{'*' * 50}")
            for subtask in subtasks:
                if self.verbose:
                    print(f"{'&' * 30}\nBeginning subtask: {subtask}\n{'&' * 30}")
                if type(self.phase_models) is dict:
                    if subtask in self.phase_models:
                        self.set_model(self.phase_models[subtask])
                    else:
                        self.set_model(DEFAULT_LLM_BACKEND)
                if (
                    subtask not in self.phase_status or not self.phase_status[subtask]
                ) and subtask == "literature review":
                    repeat = True
                    while repeat:
                        repeat = self.literature_review()
                    self.phase_status[subtask] = True
                if (
                    subtask not in self.phase_status or not self.phase_status[subtask]
                ) and subtask == "plan formulation":
                    repeat = True
                    while repeat:
                        repeat = self.plan_formulation()
                    self.phase_status[subtask] = True
                if (
                    subtask not in self.phase_status or not self.phase_status[subtask]
                ) and subtask == "data preparation":
                    repeat = True
                    while repeat:
                        repeat = self.data_preparation()
                    self.phase_status[subtask] = True
                if (
                    subtask not in self.phase_status or not self.phase_status[subtask]
                ) and subtask == "running experiments":
                    repeat = True
                    while repeat:
                        repeat = self.running_experiments()
                    self.phase_status[subtask] = True
                if (
                    subtask not in self.phase_status or not self.phase_status[subtask]
                ) and subtask == "results interpretation":
                    repeat = True
                    while repeat:
                        repeat = self.results_interpretation()
                    self.phase_status[subtask] = True
                if (
                    subtask not in self.phase_status or not self.phase_status[subtask]
                ) and subtask == "report writing":
                    repeat = True
                    while repeat:
                        repeat = self.report_writing()
                    self.phase_status[subtask] = True
                if (
                    subtask not in self.phase_status or not self.phase_status[subtask]
                ) and subtask == "report refinement":
                    return_to_exp_phase = self.report_refinement()

                    if not return_to_exp_phase:
                        if self.save:
                            self.save_state(subtask)
                        return

                    self.set_agent_attr("second_round", return_to_exp_phase)
                    self.set_agent_attr("prev_report", copy(self.phd.report))
                    self.set_agent_attr("prev_exp_results", copy(self.phd.exp_results))
                    self.set_agent_attr(
                        "prev_results_code", copy(self.phd.results_code)
                    )
                    self.set_agent_attr(
                        "prev_interpretation", copy(self.phd.interpretation)
                    )

                    self.phase_status["plan formulation"] = False
                    self.phase_status["data preparation"] = False
                    self.phase_status["running experiments"] = False
                    self.phase_status["results interpretation"] = False
                    self.phase_status["report writing"] = False
                    self.phase_status["report refinement"] = False
                    self.perform_research()
                if self.save:
                    self.save_state(subtask)
                # Calculate and print the duration of the phase
                phase_end_time = time.time()
                phase_duration = phase_end_time - phase_start_time
                print(f"Subtask '{subtask}' completed in {phase_duration:.2f} seconds.")
                self.statistics_per_phase[subtask]["time"] = phase_duration

    def report_refinement(self):
        """
        Perform report refinement phase
        @return: (bool) whether to repeat the phase
        """
        reviews = self.reviewers.inference(self.phd.plan, self.phd.report)
        print("Reviews:", reviews)
        if self.human_in_loop_flag["report refinement"]:
            print(f"Provided are reviews from a set of three reviewers: {reviews}")
            input(
                "Would you like to be completed with the project or should the agents go back and improve their experimental results?\n (y) for go back (n) for complete project: "
            )
        else:
            review_prompt = f"Provided are reviews from a set of three reviewers: {reviews}. Would you like to be completed with the project or do you want to go back to the planning phase and improve your experiments?\n Type y and nothing else to go back, type n and nothing else for complete project."
            self.phd.phases.append("report refinement")
            if self.review_override:
                if self.review_total_steps == self.review_ovrd_steps:
                    response = "n"
                else:
                    response = "y"
                    self.review_ovrd_steps += 1
            else:
                response = self.phd.inference(
                    research_topic=self.research_topic,
                    phase="report refinement",
                    feedback=review_prompt,
                    step=0,
                )
            if len(response) == 0:
                raise Exception("Model did not respond")
            response = response.lower().strip()[0]
            if response == "n":
                if verbose:
                    print("*" * 40, "\n", "REVIEW COMPLETE", "\n", "*" * 40)
                return False
            elif response == "y":
                self.set_agent_attr(
                    "reviewer_response",
                    f"Provided are reviews from a set of three reviewers: {reviews}.",
                )
                return True
            else:
                raise Exception("Model did not respond")

    def report_writing(self):
        """
        Perform report writing phase
        @return: (bool) whether to repeat the phase
        """
        # experiment notes
        report_notes = [
            _note["note"]
            for _note in self.ml_engineer.notes
            if "report writing" in _note["phases"]
        ]
        report_notes = (
            f"Notes for the task objective: {report_notes}\n"
            if len(report_notes) > 0
            else ""
        )

        self.reference_papers = []
        # instantiate mle-solver
        solver = PaperSolver(
            notes=report_notes,
            max_steps=self.papersolver_max_steps,
            plan=self.phd.plan,
            exp_code=self.phd.results_code,
            exp_results=self.phd.exp_results,
            insights=self.phd.interpretation,
            lit_review=self.phd.lit_review,
            ref_papers=self.reference_papers,
            topic=self.research_topic,
            openai_api_key=self.openai_api_key,
            llm_str=self.model_backbone["report writing"],  # type: ignore
            compile_pdf=self.compile_pdf,
        )
        # run initialization for solver
        solver.initial_solve()
        # run solver for N mle optimization steps
        for _ in range(self.papersolver_max_steps):
            solver.solve()
        # get best report results
        report = "\n".join(solver.best_report[0][0])
        score = solver.best_report[0][1]
        if self.verbose:
            print(f"Report writing completed, reward function score: {score}")
        if self.human_in_loop_flag["report writing"]:
            retry = self.human_in_loop("report writing", report)
            if retry:
                return retry
        self.set_agent_attr("report", report)
        readme = self.professor.generate_readme()
        save_to_file("./research_dir", "readme.md", readme)
        save_to_file("./research_dir", "report.txt", report)
        self.reset_agents()
        return False

    def results_interpretation(self):
        """
        Perform results interpretation phase
        @return: (bool) whether to repeat the phase
        """
        max_tries = self.max_steps
        dialogue = str()
        # iterate until max num tries to complete task is exhausted
        for _i in range(max_tries):
            resp = self.postdoc.inference(
                self.research_topic,
                "results interpretation",
                feedback=dialogue,
                step=_i,
            )
            if self.verbose:
                print("Postdoc: ", resp, "\n~~~~~~~~~~~")
            dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the postdoctoral researcher: {dialogue}"
                if self.verbose:
                    print("#" * 40, "\n", "Postdoc Dialogue:", dialogue, "\n", "#" * 40)
            if "```INTERPRETATION" in resp:
                interpretation = extract_prompt(resp, "INTERPRETATION")
                if self.human_in_loop_flag["results interpretation"]:
                    retry = self.human_in_loop("results interpretation", interpretation)
                    if retry:
                        return retry
                self.set_agent_attr("interpretation", interpretation)
                # reset agent state
                self.reset_agents()
                self.statistics_per_phase["results interpretation"]["steps"] = _i
                return False
            resp = self.phd.inference(
                self.research_topic,
                "results interpretation",
                feedback=dialogue,
                step=_i,
            )
            if self.verbose:
                print("PhD Student: ", resp, "\n~~~~~~~~~~~")
            dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = (
                    f"The following is dialogue produced by the PhD student: {dialogue}"
                )
                if self.verbose:
                    print("#" * 40, "\n", "PhD Dialogue:", dialogue, "#" * 40, "\n")
        raise Exception("Max tries during phase: Results Interpretation")

    def running_experiments(self):
        """
        Perform running experiments phase
        @return: (bool) whether to repeat the phase
        """
        # experiment notes
        experiment_notes = [
            _note["note"]
            for _note in self.ml_engineer.notes
            if "running experiments" in _note["phases"]
        ]
        experiment_notes = (
            f"Notes for the task objective: {experiment_notes}\n"
            if len(experiment_notes) > 0
            else ""
        )
        # instantiate mle-solver
        solver = MLESolver(
            dataset_code=self.ml_engineer.dataset_code,
            notes=experiment_notes,
            insights=self.ml_engineer.lit_review_sum,
            max_steps=self.mlesolver_max_steps,
            plan=self.ml_engineer.plan,
            openai_api_key=self.openai_api_key,
            llm_str=self.model_backbone["running experiments"],  # type: ignore
        )
        # run initialization for solver
        solver.initial_solve()
        # run solver for N mle optimization steps
        for _ in range(self.mlesolver_max_steps - 1):
            solver.solve()
        # get best code results
        code = "\n".join(solver.best_codes[0][0])
        # regenerate figures from top code
        execute_code(code)
        score = solver.best_codes[0][1]
        exp_results = solver.best_codes[0][2]
        if self.verbose:
            print(f"Running experiments completed, reward function score: {score}")
        if self.human_in_loop_flag["running experiments"]:
            retry = self.human_in_loop("data preparation", code)
            if retry:
                return retry
        save_to_file("./research_dir/src", "run_experiments.py", code)
        self.set_agent_attr("results_code", code)
        self.set_agent_attr("exp_results", exp_results)
        # reset agent state
        self.reset_agents()
        return False

    def data_preparation(self):
        """
        Perform data preparation phase
        @return: (bool) whether to repeat the phase
        """
        max_tries = self.max_steps
        ml_feedback = str()
        ml_dialogue = str()
        swe_feedback = str()
        ml_command = str()
        hf_engine = HFDataSearch()
        # iterate until max num tries to complete task is exhausted
        for _i in range(max_tries):
            if ml_feedback != "":
                ml_feedback_in = "Feedback provided to the ML agent: " + ml_feedback
            else:
                ml_feedback_in = ""
            resp = self.sw_engineer.inference(
                self.research_topic,
                "data preparation",
                feedback=f"{ml_dialogue}\nFeedback from previous command: {swe_feedback}\n{ml_command}{ml_feedback_in}",
                step=_i,
            )
            swe_feedback = str()
            swe_dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                swe_dialogue = f"\nThe following is dialogue produced by the SW Engineer: {dialogue}\n"
                if self.verbose:
                    print(
                        "#" * 40,
                        f"\nThe following is dialogue produced by the SW Engineer: {dialogue}",
                        "\n",
                        "#" * 40,
                    )
            if "```SUBMIT_CODE" in resp:
                final_code = extract_prompt(resp, "SUBMIT_CODE")
                code_resp = execute_code(final_code, timeout=60)
                if self.verbose:
                    print("!" * 100, "\n", f"CODE RESPONSE: {code_resp}")
                swe_feedback += f"\nCode Response: {code_resp}\n"
                if "[CODE EXECUTION ERROR]" in code_resp:
                    swe_feedback += "\nERROR: Final code had an error and could not be submitted! You must address and fix this error.\n"
                else:
                    if self.human_in_loop_flag["data preparation"]:
                        retry = self.human_in_loop("data preparation", final_code)
                        if retry:
                            return retry
                    save_to_file("./research_dir/src", "load_data.py", final_code)
                    self.set_agent_attr("dataset_code", final_code)
                    # reset agent state
                    self.reset_agents()
                    self.statistics_per_phase["data preparation"]["steps"] = _i
                    return False

            if ml_feedback != "":
                ml_feedback_in = "Feedback from previous command: " + ml_feedback
            else:
                ml_feedback_in = ""
            resp = self.ml_engineer.inference(
                self.research_topic,
                "data preparation",
                feedback=f"{swe_dialogue}\n{ml_feedback_in}",
                step=_i,
            )
            # if self.verbose: print("ML Engineer: ", resp, "\n~~~~~~~~~~~")
            ml_feedback = str()
            ml_dialogue = str()
            ml_command = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                ml_dialogue = f"\nThe following is dialogue produced by the ML Engineer: {dialogue}\n"
                if self.verbose:
                    print(
                        "#" * 40,
                        f"\nThe following is dialogue produced by the ML Engineer: {dialogue}",
                        "#" * 40,
                        "\n",
                    )
            if "```python" in resp:
                code = extract_prompt(resp, "python")
                code = self.ml_engineer.dataset_code + "\n" + code
                code_resp = execute_code(code, timeout=120)
                ml_command = f"Code produced by the ML agent:\n{code}"
                ml_feedback += f"\nCode Response: {code_resp}\n"
                if self.verbose:
                    print("!" * 100, "\n", f"CODE RESPONSE: {code_resp}")
            if "```SEARCH_HF" in resp:
                hf_query = extract_prompt(resp, "SEARCH_HF")
                hf_res = "\n".join(
                    hf_engine.results_str(hf_engine.retrieve_ds(hf_query))
                )
                ml_command = f"HF search command produced by the ML agent:\n{hf_query}"
                ml_feedback += f"Huggingface results: {hf_res}\n"
        raise Exception("Max tries during phase: Data Preparation")

    def plan_formulation(self):
        """
        Perform plan formulation phase
        @return: (bool) whether to repeat the phase
        """
        max_tries = self.max_steps
        dialogue = str()
        # iterate until max num tries to complete task is exhausted
        for _i in range(max_tries):
            # inference postdoc to
            resp = self.postdoc.inference(
                self.research_topic, "plan formulation", feedback=dialogue, step=_i
            )
            if self.verbose:
                print("Postdoc: ", resp, "\n~~~~~~~~~~~")
            dialogue = str()

            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = f"The following is dialogue produced by the postdoctoral researcher: {dialogue}"
                if self.verbose:
                    print("#" * 40, "\n", "Postdoc Dialogue:", dialogue, "\n", "#" * 40)

            if "```PLAN" in resp:
                plan = extract_prompt(resp, "PLAN")
                if self.human_in_loop_flag["plan formulation"]:
                    retry = self.human_in_loop("plan formulation", plan)
                    if retry:
                        return retry
                self.set_agent_attr("plan", plan)
                # reset agent state
                self.reset_agents()
                self.statistics_per_phase["plan formulation"]["steps"] = _i
                return False

            resp = self.phd.inference(
                self.research_topic, "plan formulation", feedback=dialogue, step=_i
            )
            if self.verbose:
                print("PhD Student: ", resp, "\n~~~~~~~~~~~")

            dialogue = str()
            if "```DIALOGUE" in resp:
                dialogue = extract_prompt(resp, "DIALOGUE")
                dialogue = (
                    f"The following is dialogue produced by the PhD student: {dialogue}"
                )
                if self.verbose:
                    print("#" * 40, "\n", "PhD Dialogue:", dialogue, "#" * 40, "\n")
        raise Exception("Max tries during phase: Plan Formulation")

    def literature_review(self):
        """
        Perform literature review phase
        @return: (bool) whether to repeat the phase
        """
        arx_eng = ArxivSearch()
        max_tries = self.max_steps * 5  # lit review often requires extra steps
        # get initial response from PhD agent
        resp = self.phd.inference(
            self.research_topic, "literature review", step=0, temp=0.8
        )
        if self.verbose:
            print(resp, "\n~~~~~~~~~~~")
        # iterate until max num tries to complete task is exhausted
        for i in range(max_tries):
            feedback = str()

            # grab summary of papers from arxiv
            if "```SUMMARY" in resp:
                query = extract_prompt(resp, "SUMMARY")
                papers = arx_eng.find_papers_by_str(query, N=self.arxiv_num_summaries)
                feedback = f"You requested arXiv papers related to the query {query}, here was the response\n{papers}"

            # grab full text from arxiv ID
            elif "```FULL_TEXT" in resp:
                query = extract_prompt(resp, "FULL_TEXT")
                # expiration timer so that paper does not remain in context too long
                arxiv_paper = (
                    f"```EXPIRATION {self.arxiv_paper_exp_time}\n"
                    + arx_eng.retrieve_full_paper_text(query)
                    + "```"
                )
                feedback = arxiv_paper

            # if add paper, extract and add to lit review, provide feedback
            elif "```ADD_PAPER" in resp:
                query = extract_prompt(resp, "ADD_PAPER")
                feedback, text = self.phd.add_review(query, arx_eng)
                if len(self.reference_papers) < self.num_ref_papers:
                    self.reference_papers.append(text)

            # completion condition
            if len(self.phd.lit_review) >= self.num_papers_lit_review:
                # generate formal review
                lit_review_sum = self.phd.format_review()
                # if human in loop -> check if human is happy with the produced review
                if self.human_in_loop_flag["literature review"]:
                    retry = self.human_in_loop("literature review", lit_review_sum)
                    # if not happy, repeat the process with human feedback
                    if retry:
                        self.phd.lit_review = []
                        return retry
                # otherwise, return lit review and move on to next stage
                if self.verbose:
                    print(self.phd.lit_review_sum)
                # set agent
                self.set_agent_attr("lit_review_sum", lit_review_sum)
                # reset agent state
                self.reset_agents()
                self.statistics_per_phase["literature review"]["steps"] = i
                return False
            resp = self.phd.inference(
                self.research_topic,
                "literature review",
                feedback=feedback,
                step=i + 1,
                temp=0.8,
            )
            if self.verbose:
                print(resp, "\n~~~~~~~~~~~")
        raise Exception("Max tries during phase: Literature Review")

    def human_in_loop(self, phase, phase_prod):
        """
        Get human feedback for phase output
        @param phase: (str) current phase
        @param phase_prod: (str) current phase result
        @return: (bool) whether to repeat the loop
        """
        print("\n\n\n\n\n")
        print(f"Presented is the result of the phase [{phase}]: {phase_prod}")
        y_or_no = None
        # repeat until a valid answer is provided
        while y_or_no not in ["y", "n"]:
            y_or_no = (
                input(
                    "\n\n\nAre you happy with the presented content? Respond Y or N: "
                )
                .strip()
                .lower()
            )
            # if person is happy with feedback, move on to next stage
            if y_or_no == "y":
                pass
            # if not ask for feedback and repeat
            elif y_or_no == "n":
                # ask the human for feedback
                notes_for_agent = input(
                    "Please provide notes for the agent so that they can try again and improve performance: "
                )
                # reset agent state
                self.reset_agents()
                # add suggestions to the notes
                self.notes.append({"phases": [phase], "note": notes_for_agent})
                return True
            else:
                print("Invalid response, type Y or N")
        return False


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    api_key = os.getenv("OPENAI_API_KEY") or cfg.api_keys.openai
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY") or cfg.api_keys.deepseek
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") or cfg.api_keys.anthropic
    fireworks_api_key = os.getenv("FIREWORKS_API_KEY") or cfg.api_keys.fireworks
    ollama_api_key = cfg.api_keys.ollama

    if ollama_api_key:
        os.environ["OLLAMA_API_KEY"] = ollama_api_key
        api_key = ollama_api_key
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if deepseek_api_key:
        os.environ["DEEPSEEK_API_KEY"] = deepseek_api_key
        api_key = deepseek_api_key
    if anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
        api_key = anthropic_api_key
    if fireworks_api_key:
        os.environ["FIREWORKS_API_KEY"] = fireworks_api_key
        api_key = fireworks_api_key
    if cfg.api_keys.ollama:
        api_key = ollama_api_key
    if cfg.ollama_host:
        os.environ["OLLAMA_HOST"] = cfg.ollama_host
    if cfg.paper_length:
        os.environ["PAPER_SOLVER_PAPER_LENGTH"] = str(cfg.paper_length)

    if not api_key:
        raise ValueError(
            "API key must be provided via config or environment variables."
        )

    if cfg.copilot_mode or cfg.research_topic is None:
        research_topic = input(
            "Please name an experiment idea for AgentLaboratory to perform: "
        )
    else:
        research_topic = cfg.research_topic

    task_notes_LLM = []
    for task_note in cfg.task_notes:
        for note in task_note.notes:
            task_notes_LLM.append(
                {
                    "phases": task_note.phases,
                    "note": note,
                }
            )

    task_notes_LLM.append(
        {
            "phases": [
                "literature review",
                "plan formulation",
                "data preparation",
                "running experiments",
                "results interpretation",
                "report writing",
                "report refinement",
            ],
            "note": f"You should always write in the following language to converse and to write the report {cfg.language}",
        }
    )

    human_in_loop: dict[str, str] = cfg.human_in_loop
    human_in_loop = {
        phase.replace("_", " "): cfg.human_in_loop[phase] for phase in human_in_loop
    }

    agent_models: dict[str, str] = cfg.agent_models
    agent_models = {
        phase.replace("_", " "): cfg.agent_models[phase] for phase in agent_models
    }

    # Initialize or load laboratory
    if cfg.load_existing:
        if not cfg.load_existing_path:
            raise ValueError("Please provide path to load existing state.")
        with open(cfg.load_existing_path, "rb") as f:
            lab = pickle.load(f)
    else:
        lab = LaboratoryWorkflow(
            research_topic=research_topic,
            notes=task_notes_LLM,
            default_llm_backend=cfg.llm_backend,
            agent_model_backbone=agent_models,
            human_in_loop_flag=human_in_loop,
            openai_api_key=api_key,
            compile_pdf=cfg.compile_latex,
            verbose=cfg.verbose,
            print_costs=cfg.print_costs,
            num_papers_lit_review=cfg.workflow.num_papers_lit_review,
            papersolver_max_steps=cfg.workflow.papersolver_max_steps,
            mlesolver_max_steps=cfg.workflow.mlesolver_max_steps,
        )

    lab.perform_research()


if __name__ == "__main__":
    main()
