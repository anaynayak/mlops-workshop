from mlops_workshop import registry

registry.list_versions()
#  version  aliases
#       2   [champion]   ← the bad one
#       1   []

registry.set_alias("champion", 1)            # roll back — inference code is untouched
