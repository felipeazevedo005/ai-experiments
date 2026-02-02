import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from dotenv import load_dotenv
import os

load_dotenv()

db = lancedb.connect(os.path.dirname(__file__) + "/data")
func = get_registry().get("openai").create(name="text-embedding-ada-002")


class Words(LanceModel):
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()


table = db.create_table("words", schema=Words, mode="overwrite")
table.add(
    [
        {"text": "hello world"},
        {"text": "barbie"},
        {"text": "uno"},
        {"text": "goodbye world"},
        {"text": "JavaScript"},
        {"text": "Java"},
        {"text": "PHP"},
        {"text": "Python"},
    ]
)

query = "animal"
actual = table.search(query).limit(5).to_pandas()
print(actual)
