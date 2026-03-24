from ingestion.docs_loader import DocsLoader


def test_identify_feature_background_tasks():
    loader = DocsLoader.__new__(DocsLoader)
    result = loader.identify_feature("BackgroundTasks are great")
    assert result == "BackgroundTasks"
