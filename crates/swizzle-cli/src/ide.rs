//! `swizzle-rt ide-config`: zero-config Fast-Apply (RAD-1 5.04x Predicted Outputs)
//! presets for AI coding IDEs and agents (Continue.dev, Aider, Zed, Cursor, Cline).

use std::path::{Path, PathBuf};

use clap::ValueEnum;
use serde_json::json;

use crate::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Ide {
    /// Continue.dev (`~/.continue/config.json`)
    Continue,
    /// Aider (`~/.aider.conf.yml` & `.aider.model.settings.yml`)
    Aider,
    /// Zed Editor (`~/.config/zed/settings.json`)
    Zed,
    /// Cursor / OpenAI-compatible Base URL & Fast-Apply settings
    Cursor,
    /// Cline / Roo Code VS Code extension settings
    Cline,
}

/// Prints or writes the configuration for `ide` pointing at `endpoint` and `model`.
pub fn run(ide: Ide, model: &str, host: &str, port: u16, write_out: Option<&Path>) -> Result<()> {
    let base_url = format!("http://{host}:{port}/v1");
    let content = render(ide, model, &base_url);
    if let Some(path) = write_out {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, &content)?;
        eprintln!("Wrote {} configuration to {}", label(ide), path.display());
    } else {
        println!("{content}");
    }
    Ok(())
}

fn label(ide: Ide) -> &'static str {
    match ide {
        Ide::Continue => "Continue.dev",
        Ide::Aider => "Aider",
        Ide::Zed => "Zed",
        Ide::Cursor => "Cursor",
        Ide::Cline => "Cline / Roo Code",
    }
}

/// Renders the exact configuration snippet for `ide`.
pub fn render(ide: Ide, model: &str, base_url: &str) -> String {
    match ide {
        Ide::Continue => serde_json::to_string_pretty(&json!({
            "models": [
                {
                    "title": format!("Swizzle Runtime ({model}) — 5.04x RAD-1 Fast-Apply"),
                    "provider": "openai",
                    "model": model,
                    "apiBase": base_url,
                    "apiKey": "swizzle-local",
                    "contextLength": 32768,
                    "completionOptions": {
                        "temperature": 0.0
                    },
                    "requestOptions": {
                        "extraBodyProperties": {
                            "receipt": false
                        }
                    }
                }
            ],
            "tabAutocompleteModel": {
                "title": format!("Swizzle Autocomplete ({model})"),
                "provider": "openai",
                "model": model,
                "apiBase": base_url,
                "apiKey": "swizzle-local"
            },
            "experimental": {
                "modelRoles": {
                    "apply": format!("Swizzle Runtime ({model}) — 5.04x RAD-1 Fast-Apply")
                }
            }
        }))
        .expect("json serializes"),
        Ide::Aider => format!(
            "# Aider configuration for Swizzle Runtime (RAD-1 5.04x Speculative Fast-Apply)\n\
             # Save as .aider.conf.yml in your repository or ~/.aider.conf.yml\n\
             openai-api-base: {base_url}\n\
             openai-api-key: swizzle-local\n\
             model: openai/{model}\n\
             edit-format: whole\n\
             streaming: true\n\
             \n\
             # Optional: pass existing file content as OpenAI Predicted Outputs (`prediction`)\n\
             # in .aider.model.settings.yml:\n\
             # - name: openai/{model}\n\
             #   edit_format: whole\n\
             #   extra_params:\n\
             #     temperature: 0.0\n"
        ),
        Ide::Zed => serde_json::to_string_pretty(&json!({
            "language_models": {
                "openai": {
                    "api_url": base_url,
                    "low_speed_timeout_in_seconds": 120,
                    "available_models": [
                        {
                            "name": model,
                            "display_name": format!("Swizzle Runtime ({model}) [RAD-1 5.04x]"),
                            "max_tokens": 32768
                        }
                    ]
                }
            },
            "assistant": {
                "version": "2",
                "default_model": {
                    "provider": "openai",
                    "model": model
                }
            }
        }))
        .expect("json serializes"),
        Ide::Cursor => serde_json::to_string_pretty(&json!({
            "swizzle.cursor.provider": "OpenAI-Compatible",
            "swizzle.cursor.overrideOpenAIBaseUrl": base_url,
            "swizzle.cursor.apiKey": "swizzle-local",
            "swizzle.cursor.model": model,
            "swizzle.cursor.fastApplyPredictedOutputs": {
                "prediction": {
                    "type": "content",
                    "content": "<existing_file_content>"
                }
            }
        }))
        .expect("json serializes"),
        Ide::Cline => serde_json::to_string_pretty(&json!({
            "apiProvider": "openai",
            "openAiBaseUrl": base_url,
            "openAiApiKey": "swizzle-local",
            "openAiModelId": model,
            "openAiStreamingEnabled": true,
            "modelInfo": {
                "contextWindow": 32768,
                "supportsPromptCache": true
            }
        }))
        .expect("json serializes"),
    }
}

/// Default config path on disk for `--write` when no file is given.
pub fn default_path(ide: Ide) -> PathBuf {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    match ide {
        Ide::Continue => home.join(".continue").join("config.json"),
        Ide::Aider => PathBuf::from(".aider.conf.yml"),
        Ide::Zed => home.join(".config").join("zed").join("settings.json"),
        Ide::Cursor => PathBuf::from("swizzle-cursor.json"),
        Ide::Cline => PathBuf::from("swizzle-cline.json"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_all_ide_presets_with_endpoint_and_model() {
        for ide in [Ide::Continue, Ide::Aider, Ide::Zed, Ide::Cursor, Ide::Cline] {
            let text = render(ide, "Qwen2.5-7B-Instruct-Q4_K_M.gguf", "http://127.0.0.1:8080/v1");
            assert!(text.contains("http://127.0.0.1:8080/v1"));
            assert!(text.contains("Qwen2.5-7B-Instruct-Q4_K_M.gguf"));
        }
    }
}
