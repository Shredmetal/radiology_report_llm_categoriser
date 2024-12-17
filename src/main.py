import os

from src.csv_handler.csv_handler import CSVHandler
from src.llm_manager.llm_config import LLMConfig
from src.llm_manager.llm_factory import LLMFactory
from src.llm_manager.llm_provider import LLMProvider
from src.llm_tasks.RadiologyReportCategoriser import RadiologyReportCategoriser

if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))

    radiology_reports_csv_path = os.path.join(current_dir, "..", "data_source", "ReportsDATASET.csv")
    save_path = os.path.join(current_dir, "..", "data_output", "ReportsDATASET_processed.csv")

    reports = CSVHandler.read_csv(radiology_reports_csv_path)

    llm_config = LLMConfig(
        provider=LLMProvider.GEMMA,
        temperature=0.0,
        max_tokens=8,
        gpu_layer_offload_count=47
    )

    llm = LLMFactory.create_llm(llm_config)

    report_categoriser = RadiologyReportCategoriser(llm)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for report in reports:
        categorised_report = report_categoriser.categorise_report(report)
        print(categorised_report)
        CSVHandler.write_to_csv(categorised_report, save_path)

