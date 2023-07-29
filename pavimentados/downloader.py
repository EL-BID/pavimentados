import logging
import os

# import tarfile
import tempfile
from pathlib import Path
from urllib import parse, request

# import boto3

logger = logging.getLogger()
pavimentados_path = Path(__file__).parent
models_url = "https://pavimenta2-artifacts.s3.amazonaws.com/models.tar.gz"


class Downloader:
    def __init__(self, models_path=pavimentados_path / "models"):
        """
        This class allows to download de models and other model data from the Inter-American Development Bank repositories

        Parameters
        ----------

        models_path: str (default instalation path of package)
                The route where is going to download and check the artifacts of the models
        """
        self.models_path = models_path

    def check_artifacts(self):
        """
        This function allows to check if the path for downloads exists
        """
        if not Path(self.models_path / "artifacts").is_dir():
            raise ImportError(
                "The route for the models is not present, it means that the models are not downloaded on this environment, "
                "use viasegura.download_models function to download them propertly"
            )

    def check_files(self, filePath):
        """
        This function allows to chec if an specific file exists

        Parameters
        ----------

        filePath: str
                Route of the file to be checked

        """
        if Path(filePath).is_file():
            return True
        else:
            return False

    def download(self, url=None, aws_access_key=None, signature=None, expires=None):
        """
        This function allows to dowload the corresponding packages using the route already on the created instance

        Parameters
        ----------

        url: str
                The signed url for downloading the models

        aws_access_key: str
                The aws access key id provided by the interamerican development bank to have access to the models

        signature: str
                The aws signature provided from IDB to download the models

        expires: int
                Time in seconds provided by the IDB in which the signature will expire

        """
        if url:
            self.models_path.mkdir(parents=True, exist_ok=True)
            temp_file_path = tempfile.NamedTemporaryFile(suffix=".tar.gz").name
            logger.info("Downloading models")
            try:
                request.urlretrieve(url, temp_file_path)
            except:  # noqa: E722
                raise Exception("Provided signature is invalid.")

            logger.info("Uncompressing models")
            # todo: recheck this
            # with tarfile.open(temp_file_path, mode="r:gz") as tfile:
            #     tfile.extractall(str(self.models_path))
            logger.info("Models are available")
            os.remove(temp_file_path)
        elif aws_access_key:
            self.models_path.mkdir(parents=True, exist_ok=True)
            params = {"AWSAccessKeyId": aws_access_key, "Signature": signature, "Expires": expires}
            composed_url = "{}?{}".format(models_url, parse.urlencode(params))
            temp_file_path = tempfile.NamedTemporaryFile(suffix=".tar.gz").name
            logger.info("Downloading models")
            try:
                request.urlretrieve(composed_url, temp_file_path)
            except:  # noqa: E722
                raise Exception("Provided signature is invalid.")
            logger.info("Uncompressing models")
            # todo: recheck this
            # with tarfile.open(temp_file_path, mode="r:gz") as tfile:
            #     tfile.extractall(str(self.models_path))
            logger.info("Models are available")
            os.remove(temp_file_path)
        else:
            raise NameError(
                "Must provide any valid method of download either signature or valid url, please contact IDB to obtain the proper data"
            )
