import requests
import logging
import commons.utils
import time

logger = logging.getLogger("CustomLogger")


class SemRelevance:

    def compute_similarity(self, text1, text2, retries=2):
        endpoint = "sem_relevance"
        url = f"http://localhost:{commons.utils.PORT}/{endpoint}"
        last_error = None

        for attempt in range(retries + 1):
            try:
                response = requests.post(url, json={"text1": text1, "text2": text2}, timeout=300)
                response.raise_for_status()
                data = response.json()
                similarity = data["similarity"]
                if similarity is None:
                    raise ValueError("Response JSON contained similarity=None.")
                return similarity
            except Exception as e:
                response_text = getattr(locals().get("response", None), "text", "")
                last_error = f"{e}{f' - Response: {response_text}' if response_text else ''}"
                logger.error(f"Semantic relevance request failed: {last_error}")
                if attempt < retries:
                    time.sleep(1)

        raise RuntimeError(f"Semantic relevance server failed after {retries + 1} attempts: {last_error}")