## WebArena-Verified Response Format

**IMPORTANT:** When you complete the task, you MUST call `save_agent_response()` to save your final response.

### Response Structure

```python
save_agent_response({
    "task_type": "RETRIEVE|MUTATE|NAVIGATE",
    "status": "SUCCESS|NOT_FOUND_ERROR|ACTION_NOT_ALLOWED_ERROR|PERMISSION_DENIED_ERROR|DATA_VALIDATION_ERROR|UNKNOWN_ERROR",
    "retrieved_data": [<results>] or null,
    "error_details": null or "<explanation>"
})
```

### Task Types
- **RETRIEVE**: Retrieving/extracting data is the main objective
- **MUTATE**: Creating, updating, or deleting data/state is the main objective  
- **NAVIGATE**: Navigating to show a specific page or search result is the main objective

### Status Codes
- `SUCCESS`: Task objective fully achieved
- `NOT_FOUND_ERROR`: Target entity/resource could not be located
- `ACTION_NOT_ALLOWED_ERROR`: Platform does not support the requested action
- `PERMISSION_DENIED_ERROR`: Current user lacks permission
- `DATA_VALIDATION_ERROR`: Required input data missing or invalid
- `UNKNOWN_ERROR`: Unexpected failure

### Guidelines
- For RETRIEVE tasks, `retrieved_data` must be a list (even for single items)
- For NAVIGATE/MUTATE tasks, `retrieved_data` should be null
- For non-SUCCESS status, provide `error_details` explaining what happened

**IMPORTANT: How to handle "no data found" in RETRIEVE tasks:**
- Use `{"status": "NOT_FOUND_ERROR", "retrieved_data": null}` when search returned no results
- SUCCESS means you completed the task; NOT_FOUND_ERROR means the target doesn't exist

### Examples

Successful retrieval:
```python
# Found data - use SUCCESS
save_agent_response({
    "task_type": "RETRIEVE",
    "status": "SUCCESS",
    "retrieved_data": ["item1", "item2"],
    "error_details": null
})
```

Successful navigation:
```
save_agent_response({"task_type": "NAVIGATE", "status": "SUCCESS", "retrieved_data": null, "error_details": null})
```

Not found error:
```python
# Found NO data - use NOT_FOUND_ERROR (not SUCCESS with empty list)
save_agent_response({
    "task_type": "RETRIEVE",
    "status": "NOT_FOUND_ERROR",
    "retrieved_data": null,
    "error_details": "No matching products found"
})
```
