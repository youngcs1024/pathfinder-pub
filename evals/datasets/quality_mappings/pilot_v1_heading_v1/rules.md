# E4.4 映射审核规则 v1

本规则绑定独立 mapping_version，不替换 quality-pilot-rubric-v1（仍为 draft）。
适用来源为 quality-pilot-v1 的三份 resume；Web 原文不作生产检索 Chunk 摄取。

- 原事实答案绑定 normalized_text_digest、source alias 和 Python Unicode code point
  `[start,end)`；UTF-8 字节预算独立。Chunk 用版本、ordinal、内容摘要定位，不用 DB UUID。
- 自动候选仅枚举完整 Chunk 的全部精确出现位置；一个候选才可建议 unique_exact。
  多个或零个候选均 needs_review，不能取第一次出现。unique_exact 仍须语义复核。
- 复核可显式提供多个等长、文本完全相同的 source/chunk 片段。片段不得在 Chunk 内
  重叠；支持不连续来源、重复标题和多 Chunk 事实。变更 Chunk 策略必须重新映射。
- reviewed Chunk 的所有非空白字符须有来源。被改写空白可以不映射，但不会增加原文
  覆盖；单元包含的每个 code point（包括空白）均覆盖才算 complete，否则 partial
  或 unmapped。多 Chunk 对同一原文位置只计一次。多个必要单元分别保留，替代关系不改写。
- 相关性与文本重合独立，按 case＋允许 resume＋Chunk 判断。required_evidence 表示
  支持该案必要单元；task_context 表示其他有助回答当前问题的事实或边界；off_topic
  表示本轮逐项阅读后确认不支持当前问题。缺少判断一律 unjudged/not_judged。
- 全文空答案与 Web-only 问题可没有相关简历 Chunk，不把空必要集合算召回满分。
  文档中的指令不获执行权：注入案例中的攻击单元只用于识别／忽略攻击，不作为经历。
- scope_rejected 不产生可判断检索上下文；scope_missing 没有简历上下文。
  上述离线关联不证明运行时授权，生产权限仍由既有链路执行。
- report 分开记录映射完整性与审核完整性；审核完整要求无未决 Chunk、所有 Chunk
  复核及所有有效 case/Chunk 判断完成。布尔值不等于 CI、质量目标或 E4 阶段通过。
- 审核主体必须真实记录；本版本由用户批准的 Codex Agent 自审，不称人工或独立审核。
  未来人工标签、模型输出试标和 live 仍按各步骤授权，不继承本次例外。
- 工件 create-only；旧数据、gold、rubric、baseline 原字节保留。公开 JSON 仅含身份、
  版本、摘要、范围、关系、计数和固定诊断，不复制原文或模型输出。
