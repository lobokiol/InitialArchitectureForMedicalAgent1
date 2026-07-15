# x-risk-tags Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create personal Cursor skill `x-risk-tags` that scores one sentence against fixed Chinese-zone risk tags and, for any score > 6, outputs dangerous tags plus one example rewrite.

**Architecture:** Single Markdown skill file. No scripts, no side reference files. The agent follows an embedded tag list, 0–10 rubric, threshold > 6, and fixed Chinese output templates. Safe path returns only one line; dangerous path lists hits and one example rewrite (author keeps final wording).

**Tech Stack:** Cursor Agent Skills (`SKILL.md` YAML frontmatter + Markdown body). Storage under personal skills dir used by this machine: `C:\Users\Admin\.agents\skills\`.

## Global Constraints

- Location (exact): `C:\Users\Admin\.agents\skills\x-risk-tags\SKILL.md`
- Approach: single `SKILL.md` only (no `tags.md`, no scripts)
- Rewrite mode C: risk note + one example rewrite; author decides final wording
- Safe path A: if no tag > 6, only `低风险 / 未触及危险阈值`
- Threshold: score > 6 (i.e. ≥ 7)
- Score scale: integer 0–10 per tag
- Do not emit the full per-tag score table to the user
- Exclude UI-only labels: `共享`, `取消`, `undefined`
- Do not post, call platform APIs, or hard-keyword-block
- Skill body keep under ~200 lines
- Spec: `docs/superpowers/specs/2026-07-15-x-risk-tags-design.md`

---

## File Map

| File | Responsibility |
|------|----------------|
| `C:\Users\Admin\.agents\skills\x-risk-tags\SKILL.md` | Complete skill: frontmatter, workflow, tag table, rubric, output templates, smoke examples |
| This repo: no runtime code | Skill is personal; only this plan/spec live in the medical-agent repo |

---

### Task 1: Create `x-risk-tags` SKILL.md

**Files:**
- Create: `C:\Users\Admin\.agents\skills\x-risk-tags\SKILL.md`
- Spec reference: `docs/superpowers/specs/2026-07-15-x-risk-tags-design.md`

**Interfaces:**
- Consumes: user sentence / short copy
- Produces: either safe one-liner, or dangerous-tag list + one example rewrite (exact templates below)

- [ ] **Step 1: Create the skill directory**

```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.agents\skills\x-risk-tags"
```

Expected: directory exists at `C:\Users\Admin\.agents\skills\x-risk-tags`.

- [ ] **Step 2: Write the full SKILL.md**

Write exactly this file content to `C:\Users\Admin\.agents\skills\x-risk-tags\SKILL.md`:

````markdown
---
name: x-risk-tags
description: >-
  Scores one sentence against Chinese-zone X risk tags (porn, scam, harassment,
  grey finance, VPN spam, bots, etc.). If any tag scores above 6, marks it
  dangerous and offers an example rewrite. Use when the user asks for
  x-risk-tags, pre-post risk check, or tag-based danger scoring of a sentence.
---

# x-risk-tags

Score one sentence (or short copy) against Chinese-zone moderation tags. Advise only — do not block posting or replace the author's final wording.

## When to use

- User asks for `x-risk-tags`
- Pre-post risk / tag danger check
- Pastes a sentence and asks whether it hits risk tags

## Workflow

1. Obtain the user's text. If missing, ask for one sentence or short copy first.
2. Internally score **every** tag in the Tag set from 0–10.
3. Select tags with score **> 6** (i.e. ≥ 7).
4. Branch:
   - **None** → reply with exactly the Safe output (nothing else).
   - **Any** → use the Dangerous output template.
5. Never dump the full 0–10 score table to the user.
6. Score by **intent + specificity**, not keyword collision. One sentence may hit multiple tags.

## Scoring rubric

| Range | Meaning |
|-------|---------|
| 0–2 | Unrelated, or neutral mention of a topic name |
| 3–6 | Related but legitimate discussion (news, platform critique, education) |
| 7–8 | Strong marketing / policy-violation intent or grey-industry copy → **dangerous** |
| 9–10 | Clear solicitation, commission, or instruction to violate → **dangerous** |

## Tag set

| Tag | Typical signals |
|-----|-----------------|
| 疑似色情 | Soft erotic implication, suggestive bait |
| 中度色情 | Explicit erotic content short of extreme |
| 极度色情 | Graphic / extreme pornography or sales thereof |
| 诈骗 | High-return promises, pig-butchering, fake support/invest |
| 诈骗黄推 | Adult bait used to push scams / private channels |
| 骚扰 | Harassment, threats, persistent unwanted contact |
| 币圈骚扰 | Crypto DM spam, shill, call-to-trade harassment |
| G系骚扰 | Faction-style pile-on / targeted campaign abuse |
| 谩骂 | Insults, personal attacks |
| 谩骂水军 | Coordinated insult spam / water-army script |
| 歧视 | Discrimination against protected / identity groups |
| 政治宣传 | Sloganized political propaganda / mobilization |
| 名称广告 | Hard-sell ad style via name or copy |
| 出国留学 | Study-abroad agency spam tone |
| 金融灰产 | Underground finance, fake cashflow, illegal fundraising |
| 卖烟骚扰 | Cigarette sales spam / harassment |
| 壮阳催情药 | Aphrodisiac / sexual-enhancement drug sales |
| 机场翻墙 | VPN / circumvention service sales |
| AI机器人 | Obvious bot / template spam voice |
| 黑客与刷粉 | Hacking services, fake followers / boosting sales |

Do **not** use UI-only labels as tags: `共享`, `取消`, `undefined`.

## Output templates

### Safe (no tag > 6)

```
低风险 / 未触及危险阈值
```

### Dangerous (any tag > 6)

```
危险标签：
- {标签}（{分}/10）⚠️ 危险 — {一句话理由}

示例改写：
{改写后文本}

说明：以上为降险示例，最终措辞请自行决定。
```

Rules for dangerous output:

- List every tag with score > 6; each line has tag, score, ⚠️ 危险, and one-sentence reason.
- Provide **one** example rewrite that reduces all >6 hits where practical.
- Example rewrite demonstrates risk reduction only; author keeps final wording.
- Do not post, schedule, or call platform APIs.

## Smoke examples (self-check)

| Input | Expected shape |
|-------|----------------|
| Neutral news about X creator payouts | Safe one-liner only |
| Soft erotic bait +「私聊有福利」 | Dangerous: 疑似色情 and/or 诈骗黄推 + example rewrite |
| 「机场打折月费」VPN sales | Dangerous: 机场翻墙 + example rewrite |
| Discussing that platforms ban VPN ads | Safe (critique, not sales) |
````

- [ ] **Step 3: Verify the file exists and frontmatter parses**

```powershell
Get-Item "$env:USERPROFILE\.agents\skills\x-risk-tags\SKILL.md" | Select-Object FullName, Length
Get-Content "$env:USERPROFILE\.agents\skills\x-risk-tags\SKILL.md" -TotalCount 8
```

Expected:

- `FullName` ends with `x-risk-tags\SKILL.md`
- `Length` > 0
- First lines include `---` then `name: x-risk-tags` then `description:`

- [ ] **Step 4: Confirm key constraints are present in the file**

```powershell
Select-String -Path "$env:USERPROFILE\.agents\skills\x-risk-tags\SKILL.md" -Pattern "低风险 / 未触及危险阈值","⚠️ 危险","Score > 6|> 6|＞ 6","机场翻墙","示例改写"
```

Expected: matches for safe line, dangerous marker, threshold language, at least one tag (`机场翻墙`), and `示例改写`.

- [ ] **Step 5: Commit note**

This skill lives **outside** the medical-agent git repo. Do **not** add it under the project tree.

If documenting in-repo is still needed: the plan + spec commits are sufficient. No project `git commit` for the skill file itself.

Optional local backup (only if the user explicitly asks to version skills elsewhere): skip unless requested.

---

### Task 2: Manual behavioral smoke check

**Files:**
- Read: `C:\Users\Admin\.agents\skills\x-risk-tags\SKILL.md`
- No code changes

**Interfaces:**
- Consumes: Task 1 skill
- Produces: pass/fail notes against the four smoke cases

- [ ] **Step 1: Load the skill and run four prompts in a Cursor agent chat**

Invoke `@x-risk-tags` (or ask for `x-risk-tags`) once per case:

1. `中性：X 调整创作者分成，聚合号收益下降。`
2. `性感私密照，私聊有福利，加我。`
3. `机场月费打折，稳定翻墙，联系下单。`
4. `平台最近在打机场广告，违规售卖会被砍收益。`

- [ ] **Step 2: Assert expected shapes**

| # | Pass if |
|---|---------|
| 1 | Reply is exactly (or only) `低风险 / 未触及危险阈值` |
| 2 | Contains `⚠️ 危险`, at least one of `疑似色情` / `诈骗黄推`, plus `示例改写:` and the author-choice disclaimer |
| 3 | Contains `机场翻墙`, `⚠️ 危险`, and `示例改写:` |
| 4 | Safe one-liner (critique of policy, not sales) — not marked 危险 for 机场翻墙 |

- [ ] **Step 3: Fix deviations if any**

If a case fails (e.g. case 4 wrongly flagged sales), tighten the corresponding rubric or signal wording **in** `SKILL.md` (clarify “sales vs discussing bans”), re-run the failed case only, then stop.

- [ ] **Step 4: Mark plan tasks complete**

Check off Task 1–2 boxes in `docs/superpowers/plans/2026-07-15-x-risk-tags.md` when smoke checks pass.

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| Personal path under `.agents/skills/x-risk-tags` | Task 1 |
| Single SKILL.md, no scripts / tags.md | Task 1 |
| Full tag set (20 tags), UI noise excluded | Task 1 |
| Rubric 0–10, threshold > 6 | Task 1 |
| Safe output A | Task 1 |
| Dangerous template + one example rewrite (mode C) | Task 1 |
| No full score dump | Task 1 |
| Manual smoke cases from spec | Task 2 |
| Non-goals (no APIs, no posting) | Task 1 |

## Self-review notes

- No TBD/TODO placeholders in steps.
- File path is absolute and matches locked location.
- Frontmatter description copied from spec metadata.
- Personal skill intentionally not committed into this repo.
