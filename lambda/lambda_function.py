import json
import boto3
import logging
import random
import base64
import io
import os

# Define Logger
logger = logging.getLogger()
logging.basicConfig()
logger.setLevel(logging.INFO)

sagemaker_client = boto3.client("sagemaker-runtime")

def update_image_size(prompt_dict, height, width):
    """
    Update the image size in the prompt dictionary.
    """
    prompt_dict["5"]["inputs"]["height"] = height
    prompt_dict["5"]["inputs"]["width"] = width
    return prompt_dict

def update_seed(prompt_dict, seed=None):
    """
    Update the seed value in the prompt dictionary.
    """
    prompt_dict["25"]["inputs"]["noise_seed "] = seed
    return prompt_dict


def update_prompt_text(prompt_dict, positive_prompt):
    """
    Update the prompt text in the given prompt dictionary.
    """
    prompt_dict["6"]["inputs"]["text"] = positive_prompt
    return prompt_dict

def update_lora_name(prompt_dict, lora_name):
    prompt_dict["29"]["inputs"]["lora_name"] = lora_name
    return prompt_dict

def invoke_from_prompt(prompt_file, positive_prompt, lora_name, seed=None, height=512, width=512):
    """
    Invokes the SageMaker endpoint with the provided prompt data.

    Args:
        prompt_file (str): The path to the JSON file in ./workflow/ containing the prompt data.
        positive_prompt (str): The positive prompt to be used in the prompt data.
        lora_name (str): The name of the lora to be used in the prompt data.
        seed (int, optional): The seed value for randomization. Defaults to None.

    Returns:
        dict: The response from the SageMaker endpoint.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
    """
    logger.info("Parameters: prompt_file=%s, positive_prompt=%s, lora_name=%s, seed=%s, dimensions=%dx%d", 
                prompt_file, positive_prompt, lora_name, seed, width, height)

    # read the prompt data from json file
    with open("./workflow/" + prompt_file) as prompt_file:
        prompt_text = prompt_file.read()

    prompt_dict = json.loads(prompt_text)
    prompt_dict = update_seed(prompt_dict, seed)
    prompt_dict = update_prompt_text(prompt_dict, positive_prompt)
    prompt_dict = update_image_size(prompt_dict, height, width)
    prompt_dict = update_lora_name(prompt_dict, lora_name)
    prompt_text = json.dumps(prompt_dict)

    endpoint_name = os.environ["ENDPOINT_NAME"]
    content_type = "application/json"
    accept = "*/*"
    payload = prompt_text
    logger.info("Final payload to invoke sagemaker:")
    logger.info(json.dumps(payload, indent=4))
    response = sagemaker_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType=content_type,
        Accept=accept,
        Body=payload,
    )
    return response


def lambda_handler(event: dict, context: dict):
    """
    Lambda function handler for processing events.
    """
    try:
        logger.info("Event:")
        logger.info(json.dumps(event, indent=2))
        
        # Validate event has body
        if not event.get("body"):
            raise ValueError("Missing request body")
            
        request = json.loads(event["body"])

        # Validate required parameters
        if "positive_prompt" not in request:
            raise ValueError("Missing required parameter: positive_prompt")

        prompt_file = 'lora_flux_workflow.json'
        positive_prompt = request["positive_prompt"]
        lora_name = request.get("lora_name", "flux/dm7249atlas.safetensors")
        seed = request.get("seed", 123)
        height = request.get("height", 512)
        width = request.get("width", 512)

        response = invoke_from_prompt(
            prompt_file=prompt_file,
            positive_prompt=positive_prompt,
            lora_name=lora_name,
            seed=seed,
            height=height,
            width=width,
        )

        image_data = response["Body"].read()

        return {
            "headers": {"Content-Type": response["ContentType"]},
            "statusCode": response["ResponseMetadata"]["HTTPStatusCode"],
            "body": base64.b64encode(io.BytesIO(image_data).getvalue()).decode("utf-8"),
            "isBase64Encoded": True,
        }

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in request body: {e}")
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Invalid JSON in request body"}),
        }
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return {
            "statusCode": 400,
            "body": json.dumps({"error": str(e)}),
        }
    except Exception as e:
        # Log the full error with traceback
        logger.error("Unexpected error:", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": "Internal server error",
                "details": str(e)
            }),
        }


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    event = {
        "body": "{\"positive_prompt\": \"a handsome man smiling\",\"seed\": 123, \"height\": 512, \"width\": 512, \"lora_name\": \"flux/dm7249atlas.safetensors\"}"
    }
    lambda_handler(event, None)
