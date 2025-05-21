import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import google.generativeai as genai
from flask import Flask, request, jsonify, render_template, session
from flask_session import Session
import pandas as pd
from datetime import datetime
import logging
import json
from typing import Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from bi3.table import table

# Configure logging
print(table)