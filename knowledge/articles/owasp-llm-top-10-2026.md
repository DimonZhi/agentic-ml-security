---
title: "OWASP Top 10 for LLM Applications 2026"
source: "https://genai.owasp.org/resource/owasp-genai-llm-top-10-2026/"
added: "2026-10-05"
read: "2026-10-05"
status: partial
coverage: "Страница публикации, финальный список 2026 и разделы LLM01, LLM03, LLM05 в официальном репозитории; остальные разделы не разобраны"
tags: [owasp, llm-security, prompt-injection, excessive-agency, poisoning, supply-chain]
---

# OWASP Top 10 for LLM Applications 2026

## Коротко

[OWASP GenAI LLM Top 10 2026](https://genai.owasp.org/resource/owasp-genai-llm-top-10-2026/) — риски приложений на основе LLM.
Ниже используется [финальный список 2026](https://github.com/GenAI-Security-Project/GenAI-LLM-Top10/blob/main/2026/README.md), а не черновик или редакция 2025.

## Основные идеи

| Код 2026 | Категория, краткий пересказ |
| --- | --- |
| LLM01 | Инъекции инструкций |
| LLM02 | Раскрытие чувствительных данных |
| LLM03 | Избыточные функции, права и автономность |
| LLM04 | Компрометация цепочки поставок |
| LLM05 | Отравление данных и моделей |
| LLM06 | Неограниченный расход ресурсов |
| LLM07 | Недостоверная информация |
| LLM08 | Раскрытие скрытого контекста |
| LLM09 | Слабости векторов и embeddings |
| LLM10 | Небезопасная обработка вывода модели |

Подробно прочитаны три раздела:

- [LLM01](https://github.com/GenAI-Security-Project/GenAI-LLM-Top10/blob/main/2026/final/LLM01_PromptInjection.md): инъекции поступают через документы, инструменты и память; последствия зависят от доступных действий.
- [LLM03](https://github.com/GenAI-Security-Project/GenAI-LLM-Top10/blob/main/2026/final/LLM03_ExcessiveAgency.md): функции, права и автономность ограничиваются; авторизация обеспечивается вне LLM.
- [LLM05](https://github.com/GenAI-Security-Project/GenAI-LLM-Top10/blob/main/2026/final/LLM05_DataModelPoisoning.md): poisoning затрагивает обучение, retrieval и артефакты; автоматическое переобучение требует контроля входных данных.

## Для проекта

Интерпретация: LLM01 относится к входам DS-ассистента и ревьюера;
LLM03 — к их действиям; LLM05 — к длительному влиянию испорченных данных.
Дополняет [агентный список](owasp-agentic-top-10-2026.md).

## Ограничения и вопросы

Семь категорий прочитаны только на уровне списка. Меры не проверены на нашем стенде.
Перенос рекомендаций с LLM на классический AutoML требует отдельного обоснования.
