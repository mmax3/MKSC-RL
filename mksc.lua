package.path = package.path .. ';GymBizHawk/gymbizhawk/bizhawk.lua'
local bizhawk = require('bizhawk')

-- Hardcoded from Mario Kart - Super Circuit (Europe).wch
local watches = {
  { addr=0x3C5F, size="b", vtype="b", disp="0", domain="IWRAM", name="Start Race/Lap" },
  { addr=0x3BCE, size="w", vtype="u", disp="0", domain="IWRAM", name="Checkpoint/Lap progress" },
  { addr=0x3C1E, size="b", vtype="u", disp="0", domain="IWRAM", name="Surface type" },
  { addr=0x3C56, size="w", vtype="u", disp="0", domain="IWRAM", name="Speed" },
  { addr=0x3B98, size="d", vtype="3", disp="0", domain="IWRAM", name="Position X" },
  { addr=0x3B9C, size="d", vtype="3", disp="0", domain="IWRAM", name="Position Y" },
  { addr=0x3BA2, size="b", vtype="s", disp="0", domain="IWRAM", name="Position Z" },
  { addr=0x33C0, size="b", vtype="u", disp="0", domain="IWRAM", name="On-map" },
  { addr=0x3AC4, size="b", vtype="u", disp="0", domain="IWRAM", name="Cloud" },
}

local function read_u16(addr, domain)
  if memory.read_u16_le then return memory.read_u16_le(addr, domain) end
  return memory.read_u16(addr, domain)
end

local function read_s16(addr, domain)
  if memory.read_s16_le then return memory.read_s16_le(addr, domain) end
  return memory.read_s16(addr, domain)
end

local function read_u32(addr, domain)
  if memory.read_u32_le then return memory.read_u32_le(addr, domain) end
  return memory.read_u32(addr, domain)
end

local function read_s32(addr, domain)
  if memory.read_s32_le then return memory.read_s32_le(addr, domain) end
  return memory.read_s32(addr, domain)
end

local function read_watch_value(w)
  if w.size == "b" then
    if w.vtype == "s" then return memory.read_s8(w.addr, w.domain) end
    return memory.read_u8(w.addr, w.domain)
  elseif w.size == "w" then
    if w.vtype == "s" then return read_s16(w.addr, w.domain) end
    return read_u16(w.addr, w.domain)
  elseif w.size == "d" then
    if w.vtype == "s" or w.vtype == "3" then return read_s32(w.addr, w.domain) end
    return read_u32(w.addr, w.domain)
  end
  return 0
end

local EnvProcessor = {}
EnvProcessor.new = function()
    local this = {}
    this.NAME = "MKSC"
    this.ROM = bizhawk.getenv_safe("MKSC_PATH")
    this.HASH = ""
    this.ACTION_SPACE = {
        -- 0: A, 1: A+Left, 2: A+Right
        "int 0 2",
    }
    this.OBSERVATION_SPACE = {}

    -- Mirror RewardTracker defaults/coefficients
    this.N_CHECKPOINTS = 47
    this.MAX_CP_STEP = 2
    -- value of 60 is 1s at 60 fps
    this.T_STUCK = 120
    this.T_REVERSE = 180
    this.T_SLOW_NO_PROG = 300
    this.STUCK_RADIUS = 25.0
    this.STUCK_RADIUS2 = this.STUCK_RADIUS * this.STUCK_RADIUS
    this.SLOW_SPEED_EPS = 470 -- usually when grinding against a wall or off-track

    this.K_P = 1.0
    this.K_TIME = 0.001
    this.K_REVERSE = 0.015 -- originally 0.010
    this.K_STUCK = 0.010 -- originally 0.005
    this.K_SLOW = 0.005 -- originally 0.003
    this.K_DONE = 1.0

    this.LAP_REWARD = 5.0
    this.FINISH_REWARD = 10.0
    this.FINISH_LAP_TARGET = 4 -- MKSC has 3 laps, but we allow finishing the episode on lap 4 to ensure the "finish line crossing" is included in the reward.
    this.SURFACE_PENALTY = 0.03 -- 0.01 originally, but increased to discourage off-track grinding
    this.ON_MAP_PENALTY = 0.1

    -- Reward summary:
    -- - progress_units = (lap delta * 47) + signed checkpoint delta (per-step capped)
    -- - r_progress = 1.0 * progress_units
    -- - r_time = small per-step penalty (time cost) to avoid "do nothing" / wandering local optima
    -- - r_reverse = small per-step penalty while reverse counter is active (progress_units<0 recently)
    -- - r_stuck = small per-step penalty while stuck counter is active (progress_units==0 and XY within radius)
    -- - r_slow = small per-step penalty while slow_no_progress counter is active (progress_units==0 and speed is low)
    -- - r_lap = bonus for each completed lap transition (dlap > 0)
    -- - r_finish = bonus when finish lap target is reached (also forces done)
    -- - r_surface = per-step penalty when surface != 0
    -- - r_on_map = per-step penalty when on_map != 1
    -- - r_done = terminal penalty applied only on done=true (stuck/reverse/slow_no_progress)
    -- - r_total = r_progress + r_lap + r_finish + r_time + r_reverse + r_stuck + r_slow + r_surface + r_on_map + r_done
    -- Termination:
    -- - reverse_steps >= T_reverse OR stuck_steps >= T_stuck OR slow_no_prog_steps >= T_slow_no_prog
    --   OR lap >= finish_lap_target

    this.setup = function(self, env, setup_str)
        self.env = env
        -- On direct/debug reruns (luaRunCount > 1), observation_type may be unset.
        if self.env.observation_type == nil or self.env.observation_type == "" then
            self.env.observation_type = "IMAGE"
        end
        if self.env.observation_type ~= "IMAGE" then
            error("observation_type must be IMAGE")
        end
        self.stats_update_every = 5
        self.stats_form = nil
        self.stats_labels = {}
    end

    this._readTrackState = function(self)
        -- 0x3C5F bit layout:
        -- - MSB (bit 7): race active flag (set after crossing start line)
        -- - low 2 bits: lap index (0..3), where active 0->lap1, 1->lap2, 2->lap3, 3->lap4(finish)
        local lap_raw = memory.read_u8(0x3C5F, "IWRAM")
        local lap = 0
        if lap_raw >= 0x80 then
            local lap_idx = lap_raw % 4
            lap = lap_idx + 1
        end
        local cp_raw = read_u16(0x3BCE, "IWRAM")
        local cp = cp_raw % self.N_CHECKPOINTS
        local speed = read_u16(0x3C56, "IWRAM")
        local surface = memory.read_u8(0x3C1E, "IWRAM")
        local on_map = memory.read_u8(0x33C0, "IWRAM")
        local x = read_s32(0x3B98, "IWRAM")
        local y = read_s32(0x3B9C, "IWRAM")
        local x_pos = x / 65536.0
        local y_pos = y / 65536.0
        return lap, cp, speed, surface, on_map, x, y, x_pos, y_pos, lap_raw
    end

    this.reset = function(self)
        --client.reboot_core()

        -- Load prepared race state from savestate slot 1.
        savestate.loadslot(1)
        emu.frameadvance()
        local keys = self.env:getKeys()
        local key_names = {}
        for k, _ in pairs(keys) do
            table.insert(key_names, k)
        end
        table.sort(key_names)
        self.available_keys = table.concat(key_names, ",")

        self.prev_cp = nil
        self.prev_lap = nil
        self.stuck_steps = 0
        self.reverse_steps = 0
        self.slow_no_prog_steps = 0
        self.stuck_anchor_x = nil
        self.stuck_anchor_y = nil

        self.last_reward = 0.0
        self.last_done_reason = ""
        self.last_progress_units = 0
        self.last_dc_signed = 0
        self.last_lap = 0
        self.last_lap_raw = 0
        self.last_cp = 0
        self.last_speed = 0
        self.last_surface = 0
        self.last_on_map = 0

        if self.env.mode ~= "FAST_RUN" then
            self:_displayDraw()
        end
    end


    this._ensureStatsPanel = function(self)
        if self.stats_form ~= nil then
            return
        end
        self.stats_form = forms.newform(230, 165, "MKSC Stats")
        self.stats_labels["lap"] = forms.label(self.stats_form, "lap: 0", 8, 10, 210, 18)
        self.stats_labels["cp"] = forms.label(self.stats_form, "cp: 0", 8, 30, 210, 18)
        self.stats_labels["r"] = forms.label(self.stats_form, "r: 0.000", 8, 50, 210, 18)
        self.stats_labels["stuck"] = forms.label(self.stats_form, "stuck: 0", 8, 70, 210, 18)
        self.stats_labels["rev"] = forms.label(self.stats_form, "rev: 0", 8, 90, 210, 18)
        self.stats_labels["slow"] = forms.label(self.stats_form, "slow: 0", 8, 110, 210, 18)
    end

    this._displayDraw = function(self)
        self:_ensureStatsPanel()
        forms.settext(self.stats_labels["lap"], "lap: " .. tostring(self.last_lap))
        forms.settext(self.stats_labels["cp"], "cp: " .. tostring(self.last_cp))
        forms.settext(self.stats_labels["r"], "r: " .. string.format("%.3f", self.last_reward))
        forms.settext(self.stats_labels["stuck"], "stuck: " .. tostring(self.stuck_steps))
        forms.settext(self.stats_labels["rev"], "rev: " .. tostring(self.reverse_steps))
        forms.settext(self.stats_labels["slow"], "slow: " .. tostring(self.slow_no_prog_steps))
    end

    this._computeRewardFromTrackState = function(self, lap, cp, speed, surface, on_map, x_pos, y_pos, lap_raw)
        local done = false
        local done_reason = ""
        local progress_units = 0
        local dlap = 0
        local dc_signed = 0
        -- counters
        -- reward breakdown
        -- termination flags (for sanity tests)

        -- ---- derived progress + reward state (single track) ----
        -- Plausibility filter for checkpoint deltas per step.
        -- If you sample/act every Nth frame (collect_every > 1), the checkpoint can advance
        -- by more than 1-2 between sampled steps. A too-small value will zero-out legitimate
        -- progress, which can lead to local optima like "drive in circles".

        -- Reward coefficients:
        -- - k_p: weight for progress_units (main learning signal).
        --   Higher -> stronger incentive to advance checkpoints/laps.
        -- - k_time: small per-step penalty (time cost).
        -- - k_reverse: per-step penalty while reverse timer is active.
        -- - k_stuck: per-step penalty while stuck timer is active.
        -- - k_slow: per-step penalty while slow_no_progress timer is active.
        -- - k_done: one-time terminal penalty when done=true.

        -- wall grind / off-track speed cap
        -- display-space XY units; tune if too strict/loose

        -- ---- LAP 0 SHORT-CIRCUIT ----
        -- Only short-circuit before race starts. MKSC may keep lap==0 while already driving.
        if lap == 0 and (lap_raw ~= nil and lap_raw < 0x80) then
            self.prev_cp = cp
            self.prev_lap = lap
            self.reverse_steps = 0
            self.slow_no_prog_steps = 0

            if self.stuck_anchor_x == nil or self.stuck_anchor_y == nil then
                self.stuck_anchor_x = x_pos
                self.stuck_anchor_y = y_pos
                self.stuck_steps = 1
            else
                local dx = x_pos - self.stuck_anchor_x
                local dy = y_pos - self.stuck_anchor_y
                if (dx * dx + dy * dy) <= self.STUCK_RADIUS2 then
                    self.stuck_steps = self.stuck_steps + 1
                else
                    self.stuck_steps = 0
                    self.stuck_anchor_x = x_pos
                    self.stuck_anchor_y = y_pos
                end
            end

            self.last_reward = 0.0
            self.last_done_reason = ""
            self.last_progress_units = 0
            self.last_dc_signed = 0
            return 0.0, false, false
        end

        if self.prev_cp == nil then
            self.prev_cp = cp
            self.prev_lap = lap
            self.stuck_steps = 0
            self.stuck_anchor_x = nil
            self.stuck_anchor_y = nil
            self.reverse_steps = 0
            self.slow_no_prog_steps = 0
            self.last_reward = 0.0
            self.last_done_reason = ""
            self.last_progress_units = 0
            self.last_dc_signed = 0
            return 0.0, false, false
        end

        local dc_fwd = (cp - self.prev_cp) % self.N_CHECKPOINTS
        local dc_back = (self.prev_cp - cp) % self.N_CHECKPOINTS

        -- signed checkpoint delta with plausibility filter
        if dc_fwd >= 1 and dc_fwd <= self.MAX_CP_STEP then
            dc_signed = dc_fwd
        elseif dc_back >= 1 and dc_back <= self.MAX_CP_STEP then
            dc_signed = -dc_back
        else
            dc_signed = 0
        end

        local dlap_raw = lap - self.prev_lap
        if dlap_raw < 0 then
            dlap_raw = 0
        elseif dlap_raw > 1 then
            dlap_raw = 1
        end

        -- Treat race-start transition (lap 0 -> 1) as initialization, not progress reward.
        if self.prev_lap == 0 and lap == 1 then
            dlap_raw = 0
            dc_signed = 0
        end
        dlap = dlap_raw

        -- asymmetric checkpoint reward:
        -- forward checkpoint +1x, backward checkpoint -2x
        local cp_progress = 0
        if dc_signed > 0 then
            cp_progress = dc_signed
        elseif dc_signed < 0 then
            -- dc_signed is negative -> larger negative penalty
            cp_progress = 2 * dc_signed
        end
        progress_units = dlap * self.N_CHECKPOINTS + cp_progress

        -- reverse counter (still based on progress)
        if progress_units < 0 then
            self.reverse_steps = self.reverse_steps + 1
        elseif progress_units > 0 then
            self.reverse_steps = 0
        elseif self.reverse_steps > 0 then
            self.reverse_steps = self.reverse_steps + 1
        end

        -- stuck counter: no progress + staying in small XY radius
        if progress_units > 0 then
            self.stuck_steps = 0
            self.stuck_anchor_x = nil
            self.stuck_anchor_y = nil
        else
            if self.stuck_anchor_x == nil or self.stuck_anchor_y == nil then
                self.stuck_anchor_x = x_pos
                self.stuck_anchor_y = y_pos
                self.stuck_steps = 1
            else
                local dx = x_pos - self.stuck_anchor_x
                local dy = y_pos - self.stuck_anchor_y
                if (dx * dx + dy * dy) <= self.STUCK_RADIUS2 then
                    self.stuck_steps = self.stuck_steps + 1
                else
                    self.stuck_steps = 0
                    self.stuck_anchor_x = x_pos
                    self.stuck_anchor_y = y_pos
                end
            end
        end

        -- slow no-progress counter: wall grind / off-track
        if progress_units == 0 and speed <= self.SLOW_SPEED_EPS then
            self.slow_no_prog_steps = self.slow_no_prog_steps + 1
        else
            self.slow_no_prog_steps = 0
        end

        -- termination conditions
        if self.reverse_steps >= self.T_REVERSE then
            done = true
            done_reason = "reverse"
        elseif self.stuck_steps >= self.T_STUCK then
            done = true
            done_reason = "stuck"
        elseif self.slow_no_prog_steps >= self.T_SLOW_NO_PROG then
            done = true
            done_reason = "slow_no_progress"
        elseif lap >= self.FINISH_LAP_TARGET then
            done = true
            done_reason = "race_finished"
        end

        -- Print only on episode end to avoid spamming the console.
        -- Keep this intentionally minimal: done reason + key progress fields.
        if done then
            print(
                "MKSC done_reason=" .. done_reason ..
                " lap=" .. tostring(lap) ..
                " cp=" .. tostring(cp) ..
                " progress_units=" .. tostring(progress_units)
            )
        end

        -- reward breakdown (simplified)
        local r_progress = self.K_P * progress_units
        local r_lap = (dlap > 0) and (self.LAP_REWARD * dlap) or 0.0
        local r_finish = (done_reason == "race_finished") and self.FINISH_REWARD or 0.0
        local r_time = -self.K_TIME
        local r_reverse = (self.reverse_steps > 0) and (-self.K_REVERSE) or 0.0
        local r_stuck = (self.stuck_steps > 0) and (-self.K_STUCK) or 0.0
        local r_slow = (self.slow_no_prog_steps > 0) and (-self.K_SLOW) or 0.0
        local r_surface = (surface ~= 0) and (-self.SURFACE_PENALTY) or 0.0
        local r_on_map = (on_map ~= 1) and (-self.ON_MAP_PENALTY) or 0.0
        local r_done = done and (-self.K_DONE) or 0.0
        local r_total = r_progress + r_lap + r_finish + r_time + r_reverse + r_stuck + r_slow + r_surface + r_on_map + r_done

        self.prev_cp = cp
        self.prev_lap = lap
        self.last_reward = r_total
        self.last_done_reason = done_reason
        self.last_progress_units = progress_units
        self.last_dc_signed = dc_signed

        return r_total, done, false
    end


    this.step = function(self, action)
        if action ~= nil then
            local keys = {}
            local act = tonumber(action[1]) or 0
            -- Always accelerate (A); optionally add steering.
            table.insert(keys, "A")
            if act == 1 then
                table.insert(keys, "Left")
            elseif act == 2 then
                table.insert(keys, "Right")
            end
            self.env:setKeys(keys)
        end
        emu.frameadvance()

        local lap, cp, speed, surface, on_map, _, _, x_pos, y_pos, lap_raw = self:_readTrackState()
        self.last_lap = lap
        self.last_lap_raw = lap_raw
        self.last_cp = cp
        self.last_speed = speed
        self.last_surface = surface
        self.last_on_map = on_map

        local r_total, terminated, truncated = self:_computeRewardFromTrackState(lap, cp, speed, surface, on_map, x_pos, y_pos, lap_raw)
        if self.env.mode ~= "FAST_RUN" and (emu.framecount() % self.stats_update_every == 0) then
            self:_displayDraw()
        end
        return r_total, terminated, truncated
    end

    this.backup = function(self)
        local d = {}
        d["prev_cp"] = self.prev_cp
        d["prev_lap"] = self.prev_lap
        d["stuck_steps"] = self.stuck_steps
        d["reverse_steps"] = self.reverse_steps
        d["slow_no_prog_steps"] = self.slow_no_prog_steps
        d["stuck_anchor_x"] = self.stuck_anchor_x
        d["stuck_anchor_y"] = self.stuck_anchor_y
        d["last_reward"] = self.last_reward
        d["last_done_reason"] = self.last_done_reason
        d["last_progress_units"] = self.last_progress_units
        d["last_dc_signed"] = self.last_dc_signed
        d["last_lap"] = self.last_lap
        d["last_cp"] = self.last_cp
        return d
    end

    this.restore = function(self, d)
        self.prev_cp = d["prev_cp"]
        self.prev_lap = d["prev_lap"]
        self.stuck_steps = d["stuck_steps"]
        self.reverse_steps = d["reverse_steps"]
        self.slow_no_prog_steps = d["slow_no_prog_steps"]
        self.stuck_anchor_x = d["stuck_anchor_x"]
        self.stuck_anchor_y = d["stuck_anchor_y"]
        self.last_reward = d["last_reward"] or 0.0
        self.last_done_reason = d["last_done_reason"] or ""
        self.last_progress_units = d["last_progress_units"] or 0
        self.last_dc_signed = d["last_dc_signed"] or 0
        self.last_lap = d["last_lap"] or 0
        self.last_cp = d["last_cp"] or 0
    end


    this.getObservation = function(self)
        local d = {}
        return d
    end

    this.getInfo = function(self)
        local d = {}
        d["lap"] = self.last_lap or 0
        d["lap_raw"] = self.last_lap_raw or 0
        d["checkpoint"] = self.last_cp or 0
        d["progress_units"] = self.last_progress_units or 0
        d["dc_signed"] = self.last_dc_signed or 0
        d["stuck_steps"] = self.stuck_steps or 0
        d["reverse_steps"] = self.reverse_steps or 0
        d["slow_no_prog_steps"] = self.slow_no_prog_steps or 0
        d["done_reason"] = self.last_done_reason or ""
        d["keys"] = self.available_keys or ""
        return d
    end
    
    return this
end

-- main
bizhawk.run(EnvProcessor.new())
