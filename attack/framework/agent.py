import requests
import logging
import commons.utils

logger = logging.getLogger("CustomLogger")


class Agent:
    def __init__(self, model_name: str, max_new_tokens: int, temperature: float, top_p: float):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def request_to_server(self, endpoint: str, use_GET=True, port=None, **kwargs):
        try:
            if use_GET:
                dispatch = requests.get
            else:
                dispatch = requests.post

            kwargs.setdefault("timeout", 300)
            response = dispatch(f"http://localhost:{port if port else commons.utils.PORT}/{endpoint}", **kwargs)
            response.raise_for_status()
            response_text = response.text
            response_text = response_text.encode("utf-8").decode("unicode_escape")

            return response_text
        except requests.exceptions.HTTPError as http_err:
            error_msg = f"HTTP error occurred: {http_err} - Response: {response.text}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from http_err
        except requests.exceptions.ConnectionError:
            logger.error("Error: Could not connect to the server.")
            raise
        except requests.exceptions.Timeout:
            logger.error("Error: Request timed out.")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
        except ValueError:
            logger.error("Error: Response is not valid JSON.")
            raise
        except Exception as e:
            logger.error(e)
            raise
