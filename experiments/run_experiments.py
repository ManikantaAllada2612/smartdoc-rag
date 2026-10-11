
from smartdoc.config import QUESTIONS_PATH, RESULTS_PATH
from smartdoc.evaluation.dataset import load_evaluation_dataset
from smartdoc.evaluation.experiments import run_experiments, save_results
from smartdoc.retrieval.vector import VectorStore


def main():
    dataset = load_evaluation_dataset(QUESTIONS_PATH)

    vector_store = VectorStore()

    results = run_experiments(
        vector_store=vector_store,
        dataset=dataset,
        k=3,
    )

    save_results(results, RESULTS_PATH)

    print(f"Evaluation completed. Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
