from langchain_core.tools import tool
@tool
def get_patient_history(patient_name: str) -> str:
    """根据患者姓名查询病例历史信息"""
    pass
@tool  
def get_patient_by_id(patient_id: str) -> str:
    """根据患者ID查询病例信息"""
    pass