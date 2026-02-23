from langchain_core.tools import tool
import json
# 模拟病例数据（生产环境替换为真实数据库查询）
MOCK_PATIENTS = {
    "001": {
        "name": "张三",
        "age": 45,
        "gender": "男",
        "phone": "13800138001",
        "records": [
            {
                "record_id": "R001",
                "visit_date": "2024-01-15",
                "diagnosis": "高血压",
                "treatment": "降压药治疗",
                "doctor": "李医生"
            },
            {
                "record_id": "R002", 
                "visit_date": "2024-02-20",
                "diagnosis": "糖尿病",
                "treatment": "胰岛素治疗",
                "doctor": "王医生"
            }
        ]
    },
    # 更多病人...
}

@tool
def get_patient_history(patient_name: str) -> str:
    """
    根据患者姓名查询病例历史信息
    
    Args:
        patient_name: 患者姓名
    
    Returns:
        患者病例信息JSON字符串
    """
    # 模拟查询
    for pid, patient in MOCK_PATIENTS.items():
        if patient["name"] == patient_name:
            return json.dumps(patient, ensure_ascii=False)
    
    return json.dumps({"error": f"未找到患者 {patient_name} 的记录"})
@tool
def get_patient_by_id(patient_id: str) -> str:
    """
    根据患者ID查询病例信息
    
    Args:
        patient_id: 患者ID（如 001）
    
    Returns:
        患者病例信息JSON字符串
    """
    patient = MOCK_PATIENTS.get(patient_id)
    if patient:
        return json.dumps(patient, ensure_ascii=False)
    return json.dumps({"error": f"未找到患者ID {patient_id}"})
