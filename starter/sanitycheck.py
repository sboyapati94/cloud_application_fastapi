from os import path

import argparse
import importlib
import inspect
import os
import sys

FAIL_COLOR = "\033[91m"
OK_COLOR = "\033[92m"
WARN_COLOR = "\033[93m"


def run_sanity_check(test_dir):  # noqa: D401
    """Run a heuristic sanity check on the user's FastAPI tests."""

    # NOTE: directory check disabled deliberately; uncomment if needed
    # msg = f"No directory named {test_dir} found in {os.getcwd()}"
    # assert path.isdir(test_dir), (FAIL_COLOR + msg)

    # ------------------------------------------------------------------
    # introduction
    # ------------------------------------------------------------------
    print(
        "This script will perform a sanity test to ensure your code "
        "meets the criteria in the rubric.\n"
    )
    print(
        "Please enter the path to the file that contains your test "
        "cases for the GET() and POST() methods"
    )
    print("The path should be something like\nabc/def/test_xyz.py")
    filepath = input("> ")

    # ------------------------------------------------------------------
    # prepare imports
    # ------------------------------------------------------------------
    assert path.exists(filepath), (
        f"File {filepath} does not exist."
    )

    # Add project root to sys.path
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(filepath), "../..")
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    sys.path.insert(0, os.path.abspath(os.path.dirname(filepath)))
    sys.path.insert(
        0,
        os.path.abspath(
            os.path.join(os.path.dirname(filepath), "..")
        ),
    )

    # Convert path to module name
    module_path = os.path.relpath(filepath, project_root)
    module_name = (
        module_path.replace("/", ".")
        .replace("\\", ".")
        .rsplit(".", 1)[0]
    )
    module = importlib.import_module(module_name)

    # ------------------------------------------------------------------
    # collect test functions
    # ------------------------------------------------------------------
    test_function_names = [
        name for name in dir(module)
        if inspect.isfunction(getattr(module, name))
        and not name.startswith("__")
    ]

    test_functions_for_get = [
        name for name in test_function_names
        if ".get(" in inspect.getsource(getattr(module, name))
    ]
    test_functions_for_post = [
        name for name in test_function_names
        if ".post(" in inspect.getsource(getattr(module, name))
    ]

    print("\n============= Sanity Check Report ===========")
    SANITY_TEST_PASSING = True
    WARNING_COUNT = 1

    # ------------------------------------------------------------------
    # GET()
    # ------------------------------------------------------------------
    TEST_FOR_GET_METHOD_RESPONSE_CODE = False
    TEST_FOR_GET_METHOD_RESPONSE_BODY = False

    if not test_functions_for_get:
        print(FAIL_COLOR + f"[{WARNING_COUNT}]")
        WARNING_COUNT += 1
        print(FAIL_COLOR + "No test cases were detected for the GET() method.")
        print(
            FAIL_COLOR
            + "Please make sure you have a test case for the GET method. "
            + "This MUST test both the status code as well as the "
            + "contents of the request object.\n"
        )
        SANITY_TEST_PASSING = False
    else:
        for func in test_functions_for_get:
            source = inspect.getsource(getattr(module, func))
            if ".status_code" in source:
                TEST_FOR_GET_METHOD_RESPONSE_CODE = True
            if ".json" in source or "json.loads" in source:
                TEST_FOR_GET_METHOD_RESPONSE_BODY = True

        if not TEST_FOR_GET_METHOD_RESPONSE_CODE:
            print(FAIL_COLOR + f"[{WARNING_COUNT}]")
            WARNING_COUNT += 1
            print(
                FAIL_COLOR
                + "Your test case for GET() does not seem to be testing "
                + "the response code.\n"
            )

        if not TEST_FOR_GET_METHOD_RESPONSE_BODY:
            print(FAIL_COLOR + f"[{WARNING_COUNT}]")
            WARNING_COUNT += 1
            print(
                FAIL_COLOR
                + "Your test case for GET() does not seem to be testing "
                + "the CONTENTS of the response.\n"
            )

    # ------------------------------------------------------------------
    # POST()
    # ------------------------------------------------------------------
    TEST_FOR_POST_METHOD_RESPONSE_CODE = False
    TEST_FOR_POST_METHOD_RESPONSE_BODY = False
    COUNT_POST_METHOD_TEST_FOR_INFERENCE_RESULT = 0

    if not test_functions_for_post:
        print(FAIL_COLOR + f"[{WARNING_COUNT}]")
        WARNING_COUNT += 1
        print(FAIL_COLOR + "No test cases detected for POST() method.")
        msg = [
            "Please make sure you have TWO test cases for POST() method.\n",
            "One test case for EACH possible inference ",
            "(results/outputs) of the ML model.\n"
        ]
        print(FAIL_COLOR + "".join(msg))
        SANITY_TEST_PASSING = False
    else:
        if len(test_functions_for_post) == 1:
            print(FAIL_COLOR + f"[{WARNING_COUNT}]")
            WARNING_COUNT += 1
            print(
                FAIL_COLOR
                + "Only one test case was detected for the POST() method."
            )
            print(
                FAIL_COLOR
                + "Please make sure you have two test cases for the POST() "
                + "method.\nOne test case for EACH of the possible inferences "
                + "(results/outputs) of the ML model.\n"
            )
            SANITY_TEST_PASSING = False

        for func in test_functions_for_post:
            source = inspect.getsource(getattr(module, func))
            if ".status_code" in source:
                TEST_FOR_POST_METHOD_RESPONSE_CODE = True
            if ".json" in source or "json.loads" in source:
                TEST_FOR_POST_METHOD_RESPONSE_BODY = True
                COUNT_POST_METHOD_TEST_FOR_INFERENCE_RESULT += 1

        if not TEST_FOR_POST_METHOD_RESPONSE_CODE:
            print(FAIL_COLOR + f"[{WARNING_COUNT}]")
            WARNING_COUNT += 1
            print(
                FAIL_COLOR
                + "One or more of your test cases for POST() do not "
                + "seem to be testing the response code.\n"
            )
        if not TEST_FOR_POST_METHOD_RESPONSE_BODY:
            print(FAIL_COLOR + f"[{WARNING_COUNT}]")
            WARNING_COUNT += 1
            print(
                FAIL_COLOR
                + "One or more of your test cases for POST() do not "
                + "seem to be testing the contents of the response.\n"
            )

        if (
            len(test_functions_for_post) >= 2
            and COUNT_POST_METHOD_TEST_FOR_INFERENCE_RESULT < 2
        ):
            print(FAIL_COLOR + f"[{WARNING_COUNT}]")
            WARNING_COUNT += 1
            print(
                FAIL_COLOR
                + "You do not seem to have TWO separate test cases, "
                + "one for each possible prediction that your model "
                + "can make."
            )

    SANITY_TEST_PASSING = (
        SANITY_TEST_PASSING
        and TEST_FOR_GET_METHOD_RESPONSE_CODE
        and TEST_FOR_GET_METHOD_RESPONSE_BODY
        and TEST_FOR_POST_METHOD_RESPONSE_CODE
        and TEST_FOR_POST_METHOD_RESPONSE_BODY
        and COUNT_POST_METHOD_TEST_FOR_INFERENCE_RESULT >= 2
    )

    if SANITY_TEST_PASSING:
        print(OK_COLOR + "Your test cases look good!")

    print(
        WARN_COLOR
        + "This is a heuristic based sanity testing and cannot guarantee "
        + "the correctness of your code."
    )
    print(
        WARN_COLOR
        + "You should still check your work against the rubric to ensure "
        + "you meet the criteria."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "test_dir",
        metavar="test_dir",
        nargs="?",
        default="tests",
        help="Name of the directory that has test files."
    )
    args = parser.parse_args()
    run_sanity_check(args.test_dir)
