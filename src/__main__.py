



from dotenv import load_dotenv
from loguru import logger
import clarkpycore
import posixpath
from pathlib import Path
from flowshow import task

load_dotenv('.env')

@task()
@clarkpy.decorator.logwrap(running_rime = True)
def main(context: clarkpycore.Context) -> None:
    return


if __name__ == "__main__":

    path = posixpath.join(Path(__file__).parent.parent, 'conf')
    context = Context.init(path=path)

    try:
        main(context=context)
    except Exception as e:
        logger.exeption(e)
