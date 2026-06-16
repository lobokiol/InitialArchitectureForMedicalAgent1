from app.core.logging import logger
from app.domain.models import AppState, DiseaseDeptResult
from app.ner.disease_dept import lookup_departments


def disease_dept_node(state: AppState) -> dict:
    """疾病链：primary_disease + companion_diseases → 科室查表。"""
    logger.info(">>> Enter node: disease_dept")
    ner = state.ner_result
    if not ner or not ner.has_disease:
        logger.warning("disease_dept_node: no diseases on state")
        return {
            "disease_dept_result": DiseaseDeptResult(
                diseases=[],
                departments=[],
                route="disease",
            )
        }

    diseases = ner.all_diseases
    departments = lookup_departments(diseases)
    result = DiseaseDeptResult(
        diseases=diseases,
        departments=departments,
        route="disease",
    )
    logger.info("disease_dept_node result=%s", result)
    return {"disease_dept_result": result}
