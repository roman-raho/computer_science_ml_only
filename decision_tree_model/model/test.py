print(">>> script start", flush=True)

try:
    import pandas as pd
    print(">>> pandas imported", flush=True)
    import sklearn
    print(">>> sklearn imported", flush=True)
except Exception as e:
    import traceback
    print(">>> import failed:", e, flush=True)
    traceback.print_exc()
except:
    print("fack")