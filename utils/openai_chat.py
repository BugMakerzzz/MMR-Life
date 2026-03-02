import re
import aiohttp  # for making API calls concurrently
import asyncio  # for running API calls concurrently
import logging  # for logging rate limit warnings and other messages
import tiktoken  # for counting tokens
import time  # for sleeping after rate limit is hit
from dataclasses import dataclass, field  # for storing API inputs, outputs, and metadata
from .config import MAX_REQUESTS_PER_MINUTE, MAX_TOKENS_PER_MINUTE, REQUEST_URL, OPENAI_API_KEY

MODEL_MAP_DICT = {
    'gpt-4.1': 'gpt-4.1-2025-04-14', # 0.002
    'gpt-5': 'gpt-5-2025-08-07', # 0.00125
    'gpt-4o': 'gpt-4o-2024-11-20', # 0.0025
    'o1': 'o1-2024-12-17', # 0.015
    'o1-mini': 'o1-mini-2024-09-12', # 0.003
    'o4-mini': 'o4-mini-2025-04-16', # 0.0011
    'claude-3': 'claude-3-opus-20240229',
    'claude-3.5': 'claude-3-5-sonnet-20240620',
    'grok-2': 'grok-2'
}



async def process_api_requests(
    request_ls: list,
    results: list,
    request_url: str,
    api_key: str,
    model: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    token_encoding_name = 'cl100k_base',
    logprobs = None,
    top_logprobs = None, 
    max_tokens = None, 
    temperature = 1.0,
    sample_cnt = 1,
    max_attempts = 100,
    logging_level = 20,
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = 0.001  # 1 ms limits max throughput to 1,000 requests per second

    # initialize logging
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    # infer API endpoint and construct request header
    api_endpoint = 'chat/completions'
    request_header = {"Authorization": f"Bearer {api_key}"}

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = task_id_generator_function()  # generates integer IDs of 1, 2, 3, ...
    status_tracker = StatusTracker()  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    list_not_finished = True  # after file is empty, we'll skip reading it
    logging.debug(f"Initialization complete.")

    # initialize file reading
        # `requests` will provide requests one at a time
    requests = request_ls.__iter__()

    while True:
        # get next request (if one is not already waiting for capacity)
        if next_request is None:
            if not queue_of_requests_to_retry.empty():
                next_request = queue_of_requests_to_retry.get_nowait()
                logging.debug(f"Retrying request {next_request.task_id}: {next_request}")
            elif list_not_finished:
                try:
                    # get new request
                    request_json = {"model":model, "reasoning":{"effort": "medium"}, "messages":next(requests), "logprobs":logprobs, "top_logprobs":top_logprobs, "max_output_tokens":max_tokens, "temperature":temperature, "n":sample_cnt}
                    next_request = APIRequest(
                        task_id=next(task_id_generator),
                        request_json=request_json,
                        token_consumption=num_tokens_consumed_from_request(request_json, api_endpoint, token_encoding_name),
                        attempts_left=max_attempts,
                        metadata=request_json.pop("metadata", None)
                    )
                    status_tracker.num_tasks_started += 1
                    status_tracker.num_tasks_in_progress += 1
                    logging.debug(f"Reading request {next_request.task_id}: {next_request}")
                except StopIteration:
                    # if file runs out, set flag to stop reading it
                    logging.debug("Read file exhausted")
                    file_not_finished = False

        # update available capacity
        current_time = time.time()
        seconds_since_update = current_time - last_update_time
        available_request_capacity = min(
            available_request_capacity + max_requests_per_minute * seconds_since_update / 60.0,
            max_requests_per_minute,
        )
        available_token_capacity = min(
            available_token_capacity + max_tokens_per_minute * seconds_since_update / 60.0,
            max_tokens_per_minute,
        )
        last_update_time = current_time

        # if enough capacity available, call API
        if next_request:
            next_request_tokens = next_request.token_consumption
            if (
                available_request_capacity >= 1
                and available_token_capacity >= next_request_tokens
            ):
                # update counters
                available_request_capacity -= 1
                available_token_capacity -= next_request_tokens
                next_request.attempts_left -= 1

                # call API
                asyncio.create_task(
                    next_request.call_api(
                        request_url=request_url,
                        results=results,
                        request_header=request_header,
                        retry_queue=queue_of_requests_to_retry,
                        status_tracker=status_tracker,
                    )
                )
                next_request = None  # reset next_request to empty

        # if all tasks are finished, break
        if status_tracker.num_tasks_in_progress == 0:
            break

        # main loop sleeps briefly so concurrent tasks can run
        await asyncio.sleep(seconds_to_sleep_each_loop)

        # if a rate limit error was hit recently, pause to cool down
        seconds_since_rate_limit_error = (time.time() - status_tracker.time_of_last_rate_limit_error)
        if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
            remaining_seconds_to_pause = (seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error)
            await asyncio.sleep(remaining_seconds_to_pause)
            # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
            logging.warn(f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}")

# dataclasses

@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict
    result: list = field(default_factory=list)

    async def call_api(
        self,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        status_tracker: StatusTracker,
        results: list
    ):
        """Calls the OpenAI API and saves results."""
        logging.info(f"Starting request #{self.task_id}")
        # logging.debug(f"Starting request #{self.task_id}")
        error = None
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url=request_url, headers=request_header, json=self.request_json,
                ) as response:
                    response = await response.json()
            if "error" in response:
                logging.warning(
                    f"Request {self.task_id} failed with error {response['error']}"
                )
                status_tracker.num_api_errors += 1
                error = response
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1  # rate limit errors are counted separately

        except Exception as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}")
                data = (
                    [self.request_json, [str(e) for e in self.result], self.metadata]
                    if self.metadata
                    else [self.request_json, [str(e) for e in self.result]]
                )
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            data = (
                [self.request_json, response, self.metadata]
                if self.metadata
                else [self.request_json, response]
            )
            results.append(data)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1

# functions

def api_endpoint_from_url(request_url):
    """Extract the API endpoint from the request URL."""
    match = re.search('^https://[^/]+/v\\d+/(.+)$', request_url)
    return match[1]


def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    token_encoding_name: str,
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""
    encoding = tiktoken.get_encoding(token_encoding_name)
    # if completions request, tokens = prompt + n * max_tokens
    if api_endpoint.endswith("completions"):
        max_tokens = request_json.get("max_completion_tokens", 15)
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens

        # chat completions
        if api_endpoint.startswith("chat/"):
            num_tokens = 0
            for message in request_json["messages"]:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens -= 1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens + completion_tokens
        # normal completions
        else:
            prompt = request_json["prompt"]
            if isinstance(prompt, str):  # single prompt
                prompt_tokens = len(encoding.encode(prompt))
                num_tokens = prompt_tokens + completion_tokens
                return num_tokens
            elif isinstance(prompt, list):  # multiple prompts
                prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                num_tokens = prompt_tokens + completion_tokens * len(prompt)
                return num_tokens
            else:
                raise TypeError('Expecting either string or list of strings for "prompt" field in completion request')
    # if embeddings request, tokens = input tokens
    elif api_endpoint == "embeddings":
        input = request_json["input"]
        if isinstance(input, str):  # single input
            num_tokens = len(encoding.encode(input))
            return num_tokens
        elif isinstance(input, list):  # multiple inputs
            num_tokens = sum([len(encoding.encode(i)) for i in input])
            return num_tokens
        else:
            raise TypeError('Expecting either string or list of strings for "inputs" field in embedding request')
    # more logic needed to support other API calls (e.g., edits, inserts, DALL-E)
    else:
        raise NotImplementedError(f'API endpoint "{api_endpoint}" not implemented in this script')


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1 

def chat_generate(requests, model, logprobs=None, top_logprobs=None, max_tokens=500, temperature=1.0, sample_cnt=1):
    model = MODEL_MAP_DICT[model]
    results = []
    asyncio.run(
        process_api_requests(
            request_ls = requests,
            request_url = REQUEST_URL,
            api_key=OPENAI_API_KEY,
            model=model,
            max_requests_per_minute=float(MAX_REQUESTS_PER_MINUTE),
            max_tokens_per_minute=float(MAX_TOKENS_PER_MINUTE),
            results=results,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_tokens=max_tokens,
            temperature=temperature,
            sample_cnt=sample_cnt
        )
    )
    return results