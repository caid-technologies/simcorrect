import sim_pair as SP, mujoco, numpy as np
model, data = SP.load_model()
mujoco.mj_forward(model, data)

print("=== ALL BODIES ===")
for i in range(model.nbody):
    print(f"  [{i}] {model.body(i).name!r:40s} pos={data.xpos[i]}")

print("\n=== ALL JOINTS ===")
for i in range(model.njnt):
    print(f"  [{i}] {model.joint(i).name!r:40s} type={model.jnt_type[i]} qpos_addr={model.jnt_qposadr[i]} dof_addr={model.jnt_dofadr[i]}")

print("\n=== ALL SITES ===")
for i in range(model.nsite):
    print(f"  [{i}] {model.site(i).name!r:40s} pos={data.site_xpos[i]}")

print("\n=== ALL ACTUATORS ===")
for i in range(model.nu):
    print(f"  [{i}] {model.actuator(i).name!r:40s} kp={model.actuator_gainprm[i][0]} ctrl={model.actuator_ctrlrange[i]}")

print("\n=== ALL EQUALITY ===")
for i in range(model.neq):
    print(f"  [{i}] {model.equality(i).name!r:40s} active={data.eq_active[i]}")

print("\n=== TABLE SURFACE ===")
for i in range(model.nbody):
    name = model.body(i).name
    if "table" in name.lower():
        bpos = data.xpos[i]
        print(f"  TABLE: {name!r} world_pos={bpos}")
        for g in range(model.ngeom):
            if model.geom_bodyid[g] == i:
                gsize = model.geom_size[g]
                gpos  = model.geom_pos[g]
                print(f"    geom={model.geom(g).name!r} size={gsize} local_pos={gpos} surface_z={bpos[2]+gpos[2]+gsize[2]:.4f}")

print("\n=== WRIST BODIES ===")
for i in range(model.nbody):
    name = model.body(i).name
    if "wrist" in name.lower():
        print(f"  {name!r} local_pos={model.body_pos[i]} world_pos={data.xpos[i]}")

print("\n=== FREEJOINT ANALYSIS ===")
for i in range(model.njnt):
    if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
        name = model.joint(i).name
        addr = model.jnt_qposadr[i]
        daddr = model.jnt_dofadr[i]
        xyz  = data.qpos[addr:addr+3]
        print(f"  FREEJOINT: {name!r} qpos_addr={addr} dof_addr={daddr} xyz={xyz}")
        for b in range(model.nbody):
            if model.body_jntadr[b] == i:
                print(f"    body: {model.body(b).name!r} world_pos={data.xpos[b]}")
