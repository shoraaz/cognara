"""One-off check: current graph extraction progress."""
from app.retrieval import graph_store

driver = graph_store.get_driver()
stats = graph_store.get_graph_stats(driver)
print("Current graph stats:", stats)
driver.close()
