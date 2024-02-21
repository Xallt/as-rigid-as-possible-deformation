def other_point(face, i, j):
    v_id = set(face)
    v_id.remove(i)
    v_id.remove(j)
    return v_id.pop()
