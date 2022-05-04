#!/bin/bash

NAME="project_settings"                                  # Name of the application
DJANGODIR=/app             # Django project directory
SOCKFILE=/app/run/gunicorn.sock  # we will communicte using this unix socket
NUM_WORKERS=3                                     # how many worker processes should Gunicorn spawn
DJANGO_SETTINGS_MODULE=project_settings.settings             # which settings file should Django use
DJANGO_WSGI_MODULE=project_settings.wsgi                     # WSGI module name

echo "Starting $NAME as `whoami`"

# Create the run directory if it doesn't exist
RUNDIR=$(dirname $SOCKFILE)
test -d $RUNDIR || mkdir -p $RUNDIR

# Start your Django Gunicorn
 gunicorn project_settings.wsgi:application --bind=unix:$SOCKFILE --workers $NUM_WORKERS --timeout 600