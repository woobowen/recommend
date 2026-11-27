"""Read-only reviewer stress test on new temporary transaction roots."""
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
from lc_rec.beauty_history_attention import prediction_store as ps

binding = {"test": "non-author CPU", "backend": "native_eager"}
assignment = [3,1,8,5,0]

def rows(row_idx):
    return {
        mode: {"row_idx":row_idx,"generation_seconds": .125,
               "shuffle":{"input_changed":False},"candidates":[{"test_token":row_idx}],
               "sorted_generation_order":[0]}
        for mode in ps.CONDITIONS
    }

def validate(row, idx, mode):
    assert idx == row["row_idx"] and mode in ps.CONDITIONS

def atom_write_before_crash(target):
    with ps.open_store(target,binding,assignment,2) as store:
        store.bind_runtime({"scope":"fixture"})
        with store.chunk(0,validate) as put:
            for i in assignment[:2]:
                put(i,rows(i))
        with store.chunk(1,validate) as put:
            put(assignment[2],rows(assignment[2]))
            os._exit(73)

def wait_child(function, *values, expected):
    child = os.fork()
    if child == 0:
        try:
            function(*values)
        except Exception as error:
            print("child error",repr(error),flush=True)
            os._exit(81)
        os._exit(0)
    _, status = os.waitpid(child,0)
    assert os.waitstatus_to_exitcode(status) == expected, status

def expect_failure(action, needle):
    try:
        action()
    except (ValueError,RuntimeError) as error:
        assert needle in str(error), str(error)
    else:
        raise AssertionError(f"expected guard failed to fire: {needle}")

def set_open(root, ident=binding, axis=assignment):
    with ps.open_store(root,ident,axis,2):
        pass

def load_changed_rejected(root):
    # Child races the open while parent holds the writer lock.
    expect_failure(lambda:set_open(root),"holds the output lock")

def crash_first_spec(target):
    actual_replace=ps.os.replace
    def intercept(src,dst,*a,**kw):
        if Path(dst).name == ps.SPEC:
            os._exit(74)
        return actual_replace(src,dst,*a,**kw)
    ps.os.replace=intercept
    with ps.open_store(target,binding,assignment,2):
        raise AssertionError("expected hard exit before specification rename")

def finish_chunk(store,n,indices):
    with store.chunk(n,validate) as put:
        for i in indices: put(i,rows(i))

result = {}
with tempfile.TemporaryDirectory(prefix="beauty-review-store-") as tmp:
    root = Path(tmp) / "midchunk"
    wait_child(atom_write_before_crash,root,expected=73)
    with ps.open_store(root,binding,assignment,2) as resumed:
        assert [x["assignment"] for x in resumed.committed(validate)] == [assignment[:2]]
        assert list((root/"attempts").glob("attempt-*/chunk-*")), "lost interrupted private chunk"
        wait_child(load_changed_rejected,root,expected=0)
        def write_duplicate():
            with resumed.chunk(1,validate) as put:
                put(assignment[2],rows(assignment[2]))
                put(assignment[2],rows(assignment[2]))
        expect_failure(write_duplicate,"next assigned row")
        assert len(resumed.committed(validate)) == 1
    with ps.open_store(root,binding,assignment,2) as second_resume:
        second_resume.bind_runtime({"scope":"fixture"})
        expect_failure(lambda:second_resume.bind_runtime({"scope":"changed runtime"}),"runtime changed")
        expect_failure(lambda:finish_chunk(second_resume,0,assignment[:2]), "replace committed")
        finish_chunk(second_resume,1,assignment[2:4])
        finish_chunk(second_resume,2,assignment[4:])
        assert [x["assignment"] for x in second_resume.committed(validate)] == [assignment[:2],assignment[2:4],assignment[4:]]
        report = {"prediction_binding":binding,"assignment_sha256":ps.canonical_sha256(assignment),"assigned_users":len(assignment)}
        second_resume.publish(report,{m:{"mode":m} for m in ps.CONDITIONS})
    before = (root/"final/completion.json").read_bytes()
    with ps.open_store(root,binding,assignment,2) as already:
        assert already.completed == report and already.attempt is None
    assert (root/"final/completion.json").read_bytes() == before
    for mode in ps.CONDITIONS:
        merged = root / "final" / mode / ps.PREDICTIONS
        assert [json.loads(s)["row_idx"] for s in merged.read_bytes().splitlines()] == assignment
    expect_failure(lambda:set_open(root,{"different":"run"}),"different run")
    expect_failure(lambda:set_open(Path(tmp)/"duplicate-axis",binding,[1,1]),"bad prediction axis")
    result["midchunk_crash_resume"] = {"crash_exit":73,"committed_rows":assignment[:2],
                                        "final_rows_each_mode":assignment,"locked_concurrent_duplicate_rejected":True,
                                        "duplicate_row_rejected":True,"new_binding_and_runtime_rejected":True,
                                        "final_receipt_sha256":hashlib.sha256(before).hexdigest()}
    pre = Path(tmp)/"pre-spec"
    wait_child(crash_first_spec,pre,expected=74)
    assert not (pre/ps.SPEC).exists()
    result["prespec_crash_artifacts"] = sorted(x.name for x in pre.iterdir())
    with ps.open_store(pre,binding,assignment,2) as clean:
        assert clean.committed(validate) == []
    result["prespec_crash_resume"] = "PASS"
result["source_sha256"] = hashlib.sha256(Path(ps.__file__).read_bytes()).hexdigest()
print(json.dumps(result,indent=2,allow_nan=False))
