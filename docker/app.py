from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/math", response_class=HTMLResponse)
async def math_operation(
    request: Request,
    operation: str = Form(...),
    num1: int = Form(...),
    num2: int = Form(...)
):

    if operation == "add":
        r = num1 + num2
        result = f"The sum of {num1} and {num2} is {r}"

    elif operation == "subtract":
        r = num1 - num2
        result = f"The subtract of {num1} and {num2} is {r}"

    elif operation == "multiply":
        r = num1 * num2
        result = f"The multiply of {num1} and {num2} is {r}"

    elif operation == "divide":
        r = num1 / num2
        result = f"The division of {num1} and {num2} is {r}"

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "result": result
        }
    )