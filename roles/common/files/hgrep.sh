hgrep () {
    history | sed 's/^ *[0-9]* *//' | egrep --color=auto "$@" | less
}
