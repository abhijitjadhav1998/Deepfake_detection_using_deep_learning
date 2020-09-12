# Steps to run Django project

## Requirements:

You can find the list of requirements in requirements.txt. Main requirements are listed below:

Python >= v3.6<br/>
Django >= v3.0

## Directory Structure

ml_app -> Directory containing code in views.py file
project_settings -> Contains Django settings and files to run in production
static -> Contains all css, js and json files (for face-api)
templates -> Template files for HTML


<b>Note:</b> Before running the project make sure you have created directories namely <strong>models, uploaded_images, uploaded_videos</strong> in the project root and that you have proper permissions to access them.

### Step 1: Create virtualenv

`python -m venv venv`

### Step 2: Activate virtualenv

### Step 3: Install requirements

`pip install requirements.txt`

### Step 4: Run project

`python manage.py runserver`

#### IMPORTANT: 

i. By default the server will run on PORT 8000 but you change that by passing the port in command line argument. 

ii. Change the DEBUG value to False present <strong>project_settings -> settings.py </strong> before putting into production.

iii. Add your IP/Domain to ALLOWED_HOSTS in settings.py when it is deployed to a server.

For e.g. if you deploy it to a machine with IP 8.8.8.8 & the domain assigned to that machine is example.com then it should look like this

`ALLOWED_HOSTS = ['8.8.8.8', 'example.com']`

<b>Note:</b> This is only required when DEBUG is False.
