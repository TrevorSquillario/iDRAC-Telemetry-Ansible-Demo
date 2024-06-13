#!/bin/bash
# Increase history size
export HISTSIZE=100000
export HISTFILESIZE=100000

# Append to history, don't overwrite it
shopt -s histappend 

# Save and reload the history after each command finishes
export PROMPT_COMMAND="history -a; history -c; history -r; $PROMPT_COMMAND"
