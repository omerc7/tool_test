from kubiya_sdk.tools import function_tool


@function_tool(
    description="Test empty req",
)
def empty_req():
    print("Works!!!")
