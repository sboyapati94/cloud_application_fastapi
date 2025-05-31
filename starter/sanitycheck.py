from os import path
import argparse
import importlib
import inspect
import sys

FAIL_COLOR = '\033[91m'
OK_COLOR = '\033[92m'
WARN_COLOR = '\033[93m'


def run_sanity_check(test_dir):
    print('Sanity test to ensure code meets rubric criteria.\n')
    print('Enter path to test file for GET() and POST() methods:')
    print('Format: abc/def/test_xyz.py')
    filepath = input('> ')

    assert path.exists(filepath), f"File {filepath} does not exist."
    sys.path.append(path.dirname(filepath))

    module_name = path.splitext(path.basename(filepath))[0]
    module = importlib.import_module(module_name)

    test_function_names = [
        name for name in dir(module)
        if inspect.isfunction(getattr(module, name)) and
        not name.startswith('__')
    ]

    test_functions_for_get = [
        name for name in test_function_names
        if '.get(' in inspect.getsource(getattr(module, name))
    ]

    test_functions_for_post = [
        name for name in test_function_names
        if '.post(' in inspect.getsource(getattr(module, name))
    ]

    print("\n========= Sanity Check Report =========")
    SANITY_TEST_PASSING = True
    WARNING_COUNT = 1

    # GET()
    TEST_FOR_GET_METHOD_RESPONSE_CODE = False
    TEST_FOR_GET_METHOD_RESPONSE_BODY = False
    if not test_functions_for_get:
        print(f"{FAIL_COLOR}[{WARNING_COUNT}]")
        WARNING_COUNT += 1
        print(f"{FAIL_COLOR}No GET tests found.")
        print(f"{FAIL_COLOR}Add a GET test checking status and content.\n")
        SANITY_TEST_PASSING = False

    else:
        for func in test_functions_for_get:
            source = inspect.getsource(getattr(module, func))
            if '.status_code' in source:
                TEST_FOR_GET_METHOD_RESPONSE_CODE = True
            if ('.json' in source) or ('json.loads' in source):
                TEST_FOR_GET_METHOD_RESPONSE_BODY = True

        if not TEST_FOR_GET_METHOD_RESPONSE_CODE:
            print(f"{FAIL_COLOR}[{WARNING_COUNT}]")
            WARNING_COUNT += 1
            print(f"{FAIL_COLOR}GET missing status code check.\n")

        if not TEST_FOR_GET_METHOD_RESPONSE_BODY:
            print(f"{FAIL_COLOR}[{WARNING_COUNT}]")
            WARNING_COUNT += 1
            print(f"{FAIL_COLOR}GET missing content check.\n")

    # POST()
    TEST_FOR_POST_METHOD_RESPONSE_CODE = False
    TEST_FOR_POST_METHOD_RESPONSE_BODY = False
    COUNT_POST_METHOD_TEST_FOR_INFERENCE_RESULT = 0

    if not test_functions_for_post:
        print(f"{FAIL_COLOR}[{WARNING_COUNT}]")
        WARNING_COUNT += 1
        print(f"{FAIL_COLOR}No POST tests found.")
        print(f"{FAIL_COLOR}Add two POST tests for model inferences.\n")
        SANITY_TEST_PASSING = False
    else:
        if len(test_functions_for_post) == 1:
            print(f"[{WARNING_COUNT}]")
            WARNING_COUNT += 1
            print(f"{FAIL_COLOR}Only one POST test found.")
            print(f"{FAIL_COLOR}Add two tests for model outputs.\n")
            SANITY_TEST_PASSING = False

        for func in test_functions_for_post:
            source = inspect.getsource(getattr(module, func))
            if '.status_code' in source:
                TEST_FOR_POST_METHOD_RESPONSE_CODE = True
            if ('.json' in source) or ('json.loads' in source):
                TEST_FOR_POST_METHOD_RESPONSE_BODY = True
                COUNT_POST_METHOD_TEST_FOR_INFERENCE_RESULT += 1

        if not TEST_FOR_POST_METHOD_RESPONSE_CODE:
            print(f"{FAIL_COLOR}[{WARNING_COUNT}]")
            WARNING_COUNT += 1
            print(f"{FAIL_COLOR}POST missing status code check.\n")
        if not TEST_FOR_POST_METHOD_RESPONSE_BODY:
            print(f"{FAIL_COLOR}[{WARNING_COUNT}]")
            WARNING_COUNT += 1
            print(f"{FAIL_COLOR}POST missing content check.\n")

        if (len(test_functions_for_post) >= 2 and
                COUNT_POST_METHOD_TEST_FOR_INFERENCE_RESULT < 2):
            print(f"{FAIL_COLOR}[{WARNING_COUNT}]")
            WARNING_COUNT += 1
            print(f"{FAIL_COLOR}Need two tests for model predictions.")

    SANITY_TEST_PASSING = (SANITY_TEST_PASSING and
                           TEST_FOR_GET_METHOD_RESPONSE_CODE and
                           TEST_FOR_GET_METHOD_RESPONSE_BODY and
                           TEST_FOR_POST_METHOD_RESPONSE_CODE and
                           TEST_FOR_POST_METHOD_RESPONSE_BODY and
                           COUNT_POST_METHOD_TEST_FOR_INFERENCE_RESULT >= 2)

    if SANITY_TEST_PASSING:
        print(f"{OK_COLOR}Test cases look good!")

    print(f"{WARN_COLOR}Heuristic check - verify against rubric.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'test_dir',
        metavar='test_dir',
        nargs='?',
        default='tests',
        help='Directory with test files.'
    )
    args = parser.parse_args()
    run_sanity_check(args.test_dir)
