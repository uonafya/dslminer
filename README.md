# DSL Miner

Module that containts data mining modules for the MoH DAP platform.

# Set up
 
# Setup pip (rhel based os):

    sudo yum install python-pip python-devel gcc
    or
    sudo dnf install python-pip python-devel gcc

# install and create python Virtual Environment (optional)

installation:

    sudo pip install virtualenv
    
Clone the repo:
	```
	 git clone https://github.com/uonafya/dslminer.git
	```
Creating virtual env with virtualenv tool, run the following in terminal:

    cd dslminer
    virtualenv ./

The above will create a python virtual env in your project folder.
The above copies files for your default python interpreter to the virtual environment.
If you have multiple instances of python interpreter on your machine, you can specify the version to use as:

    virtaulenv --python=/location/to/python/touse ./
    eg: virtualenv --python=/usr/local/bin/python2.7 ./

# Activate and install project dependencies:

Activate the environment as follows (if the optional virtaul env was set up otherwise skip this step):
  
    source dslminer/bin/activate

Install the projects dependencies using the requirements.txt file in dslminer by running:

    pip install -r requirements.txt


# Run the module

	```
	export FLASK_APP=api.py 
	flask run
	```
this will run the app in the default host and port: host: localhost, port: 5000

