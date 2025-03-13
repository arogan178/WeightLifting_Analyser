#!/usr/bin/env python3
"""
Weightlifting Performance Analyzer
Main application entry point
"""
import os
import sys
import argparse
from src.ui.app import launch_app

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Weightlifting Performance Analyzer')
    parser.add_argument('--no-gui', action='store_true', help='Run in command line mode without GUI')
    return parser.parse_args()

def main():
    """Main application entry point"""
    args = parse_args()
    
    # Launch the application
    launch_app(headless=args.no_gui)

if __name__ == "__main__":
    main()