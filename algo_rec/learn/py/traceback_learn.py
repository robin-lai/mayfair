import sys, traceback

def run_user_code():
    try:
        ll = []
        ll.extends()
    except Exception:
        print("Exception in user code:")
        print("-"*60)
        traceback.print_exc(file=sys.stdout)
        print("-"*60)

if __name__ == '__main__':
    run_user_code()