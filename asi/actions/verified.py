"""
WebArena-Verified specific actions.

This module contains actions specific to the WebArena-Verified evaluation
framework, including the save_agent_response action for structured output.
"""


def save_agent_response(response: dict):
    """
    Save the agent's structured response in WebArena-Verified format.

    This function saves the agent's final response to agent_response.json
    in the current working directory. The response must follow the
    WebArena-Verified schema.

    Args:
        response: A dictionary containing:
            - task_type: "RETRIEVE", "MUTATE", or "NAVIGATE"
            - status: "SUCCESS", "NOT_FOUND_ERROR", "ACTION_NOT_ALLOWED_ERROR",
                      "PERMISSION_DENIED_ERROR", "DATA_VALIDATION_ERROR", or "UNKNOWN_ERROR"
            - retrieved_data: List of results for RETRIEVE tasks, null/None otherwise
            - error_details: null/None for SUCCESS, explanation string otherwise

    Returns:
        None

    Examples:
        save_agent_response({"task_type": "RETRIEVE", "status": "SUCCESS", "retrieved_data": ["item1", "item2"], "error_details": None})
    """
    # Import inside function - necessary because browsergym exec context doesn't include module imports
    import json
    import os

    # Validate required fields
    required_fields = ["task_type", "status"]
    for field in required_fields:
        if field not in response:
            raise ValueError(f"Missing required field: {field}")

    # Validate task_type
    valid_task_types = ["RETRIEVE", "MUTATE", "NAVIGATE"]
    if response.get("task_type") not in valid_task_types:
        raise ValueError(
            f"Invalid task_type: {response.get('task_type')}. Must be one of {valid_task_types}"
        )

    # Validate status
    valid_statuses = [
        "SUCCESS",
        "NOT_FOUND_ERROR",
        "ACTION_NOT_ALLOWED_ERROR",
        "PERMISSION_DENIED_ERROR",
        "DATA_VALIDATION_ERROR",
        "UNKNOWN_ERROR",
    ]
    if response.get("status") not in valid_statuses:
        raise ValueError(
            f"Invalid status: {response.get('status')}. Must be one of {valid_statuses}"
        )

    # Ensure retrieved_data and error_details exist
    if "retrieved_data" not in response:
        response["retrieved_data"] = None
    if "error_details" not in response:
        response["error_details"] = None

    # For RETRIEVE tasks with SUCCESS, retrieved_data should be a list
    if response["task_type"] == "RETRIEVE" and response["status"] == "SUCCESS":
        if not isinstance(response["retrieved_data"], list):
            # Convert to list if not already
            if response["retrieved_data"] is not None:
                response["retrieved_data"] = [response["retrieved_data"]]

    # For non-SUCCESS status, error_details should be present (but we'll be lenient)
    # For SUCCESS status, retrieved_data should be null for NAVIGATE/MUTATE
    if response["status"] == "SUCCESS" and response["task_type"] in [
        "NAVIGATE",
        "MUTATE",
    ]:
        if response["retrieved_data"] is not None and response["retrieved_data"] != []:
            # Be lenient but log a warning
            print(
                f"[warning] retrieved_data should be null for {response['task_type']} tasks, but got: {response['retrieved_data']}"
            )

    # Save to file (prefer experiment directory when available)
    exp_dir = os.environ.get("BROWSERGYM_EXP_DIR")
    output_paths = []
    if exp_dir:
        output_paths.append(os.path.join(exp_dir, "agent_response.json"))
    output_paths.append("agent_response.json")

    for output_path in output_paths:
        with open(output_path, "w") as f:
            json.dump(response, f, indent=2, default=str)
        full_path = os.path.abspath(output_path)
        print(f"[save_agent_response] Response saved to {full_path}")
    print(
        f"[save_agent_response] Response: {json.dumps(response, indent=2, default=str)}"
    )
