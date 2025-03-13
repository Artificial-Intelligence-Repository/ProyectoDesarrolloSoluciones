FROM python:3.10

# Create user to run dash
RUN adduser --disabled-password --gecos '' dash-user

# Set working directory
WORKDIR /opt/app

# Install dependencies
COPY ./app/requirements.txt /opt/app/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Add dash-bootstrap-components to requirements.txt or install it here
RUN pip install dash-bootstrap-components

# Change ownership of the working directory
RUN chown -R dash-user:dash-user /opt/app

USER dash-user

# Expose port
EXPOSE 8000

# Command to run the app
CMD ["python", "app2.py"]