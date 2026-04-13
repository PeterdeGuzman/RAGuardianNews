import shutil
import duckdb

# Archive current model
shutil.copytree("models/bertopic_model", "models/bertopic_model_v1")

# Record the current topic assignments in a named table
conn = duckdb.connect("../guardian_articles.duckdb")
conn.execute("""
    CREATE TABLE article_topics_v1 AS 
    SELECT * FROM article_topics
""")
conn.close()

print("v1 saved")
