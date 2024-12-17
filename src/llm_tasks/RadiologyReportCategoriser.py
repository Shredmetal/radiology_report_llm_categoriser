from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any

from src.llm_tasks.pneumonia_laterality.pneumonia_laterality import PneumoniaLaterality
from src.llm_tasks.pneumonia_presence.pneumonia_presence import PneumoniaPresence
from src.llm_tasks.pneumonia_size.pneumonia_size import PneumoniaSize

class RadiologyReportCategoriser:

    def __init__(self, llm):
        self.llm = llm
        self.presence_prompt = PneumoniaPresence.get_pneumonia_presence_prompts()
        self.laterality_prompt = PneumoniaLaterality.get_pneumonia_laterality_prompts()
        self.size_prompt = PneumoniaSize.get_pneumonia_size_prompts()
        self.output_parser = StrOutputParser()

    def categorise_report(self, report: str) -> Dict[str, Any]:
        presence_chain = self.presence_prompt | self.llm | self.output_parser
        laterality_chain = self.laterality_prompt | self.llm | self.output_parser
        size_chain = self.size_prompt | self.llm | self.output_parser

        presence_result = presence_chain.invoke({"report": report}).strip().lower()

        if presence_result == "false":
            laterality_result = "not applicable"
            size_result = "not applicable"
        else:
            laterality_result = laterality_chain.invoke({"report": report}).strip().lower()
            size_result = size_chain.invoke({"report": report}).strip().lower()

        return {
            "report": report,
            "pneumonia_present": presence_result,
            "pneumonia_laterality": laterality_result,
            "pneumonia_size": size_result
        }
