# x-risk-tags Skill Design

Personal Cursor skill: take one sentence (or short copy), score against Chinese-zone risk tags, and when any tag score is greater than 6, mark it dangerous and provide an example rewrite.

## Goal

Help the author self-check X/Twitter (or similar) copy against a fixed moderation-tag list before posting. The skill advises; it does not block posting or replace the author's final wording.

## Decisions (locked)

| Decision | Choice |
|----------|--------|
| Location | Personal skill at `C:\Users\Admin\.agents\skills\x-risk-tags\SKILL.md` |
| Implementation approach | Single `SKILL.md` only (LLM rubric; no scripts, no side reference file in v1) |
| Rewrite mode | C — risk note + one example rewrite; author keeps final wording |
| Safe-path output | A — if no tag > 6, only: `低风险 / 未触及危险阈值` |
| Threshold | Score > 6 (i.e. ≥ 7) is dangerous |
| Score scale | Integer 0–10 per tag |

## Non-goals

- Posting, scheduling, or calling platform APIs
- Keyword scripts or hard blocklists
- Emitting the full 0–10 score table to the user
- UI controls from the source screenshot (`共享`, `取消`, `undefined`)

## Tag set

Exclude UI noise. Score every tag below on each run (internally):

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

## Scoring rubric

- **0–2**: Unrelated, or neutral mention of a topic name
- **3–6**: Related but legitimate discussion (news, platform critique, education)
- **7–8**: Strong marketing / policy-violation intent or grey-industry copy → dangerous
- **9–10**: Clear solicitation, commission, or instruction to violate → dangerous

Score by **intent + specificity**, not keyword collision. One sentence may hit multiple tags.

## Workflow

1. Obtain the user's text. If missing, ask for one sentence / short copy first.
2. Internally score all tags 0–10.
3. Select tags with score > 6.
4. Branch:
   - **None** → reply only: `低风险 / 未触及危险阈值`
   - **Any** → use the dangerous output template below

## Output templates

### Safe

```
低风险 / 未触及危险阈值
```

### Dangerous

```
危险标签：
- {标签}（{分}/10）⚠️ 危险 — {一句话理由}

示例改写：
{改写后文本}

说明：以上为降险示例，最终措辞请自行决定。
```

Multiple dangerous tags: list each with reason; provide **one** example rewrite that reduces all >6 hits where practical.

## Skill metadata (implementation target)

```yaml
name: x-risk-tags
description: >-
  Scores one sentence against Chinese-zone X risk tags (porn, scam, harassment,
  grey finance, VPN spam, bots, etc.). If any tag scores above 6, marks it
  dangerous and offers an example rewrite. Use when the user asks for
  x-risk-tags, pre-post risk check, or tag-based danger scoring of a sentence.
```

Prefer auto-discovery via description triggers; keep body under ~200 lines.

## Testing (manual, for plan phase)

| Input | Expected |
|-------|----------|
| Neutral platform news about creator payouts | Safe one-liner |
| Soft erotic bait +「私聊有福利」 | Dangerous: 疑似色情 and/or 诈骗黄推; example rewrite |
| 「机场打折月费」VPN sales | Dangerous: 机场翻墙; example rewrite |
| Discussing that platforms ban VPN ads | Safe or ≤6 (critique, not sales) |

## Out of scope for v1

- Progressive disclosure (`tags.md`)
- Batch file scanning
- English-first tag aliases
- Integration with `x-post` / `post-quality` (may call either later; not required here)
