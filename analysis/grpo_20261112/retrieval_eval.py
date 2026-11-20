import runpy,uvicorn
original_run=uvicorn.run
def serve(app,**kwargs):
    kwargs["port"]=8013
    return original_run(app,**kwargs)
uvicorn.run=serve
runpy.run_path("search_r1/search/retrieval_server.py",run_name="__main__")
