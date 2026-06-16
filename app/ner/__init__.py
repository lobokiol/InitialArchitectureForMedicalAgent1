from app.ner.models import EntityExtractResult, NERExtractOutput

__all__ = ["EntityExtractResult", "NERExtractOutput", "extract_entity_tags"]


def __getattr__(name: str):
    if name == "extract_entity_tags":
        from app.ner.service import extract_entity_tags

        return extract_entity_tags
    raise AttributeError(name)
