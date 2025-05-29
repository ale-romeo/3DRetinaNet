with open("get_kinetics_weights.sh", "rb") as f:
    content = f.read().replace(b"\r\n", b"\n")
with open("get_kinetics_weights.sh", "wb") as f:
    f.write(content)
