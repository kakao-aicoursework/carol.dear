import pynecone as pc

class TranslatorConfig(pc.Config):
    pass

config = TranslatorConfig(
    app_name="sink",
    db_url="sqlite:///pynecone.db",
    env=pc.Env.DEV,
)