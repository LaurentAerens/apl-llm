const vscode = require('vscode');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

let pythonProcess = null;
let requestCounter = 0;
const activeRequests = new Map();
let outputChannel = null;

function log(message) {
    if (outputChannel) {
        outputChannel.appendLine(`[${new Date().toLocaleTimeString()}] ${message}`);
    }
}

function startDaemon() {
    if (pythonProcess) {
        log('Daemon already running. Stopping it first...');
        stopDaemon();
    }

    const config = vscode.workspace.getConfiguration('apl-slm');
    const legacyConfig = vscode.workspace.getConfiguration('apl-llm');
    const pythonPath = config.get('pythonPath') || legacyConfig.get('pythonPath') || 'python';
    const modelPath = config.get('modelPath') || legacyConfig.get('modelPath') || 'checkpoints/apl_slm_best.pt';
    const serverScript = path.join(__dirname, 'autocomplete_server.py');

    // Resolve model path relative to workspace root or extension directory
    let resolvedModelPath = modelPath;
    if (!path.isAbsolute(modelPath)) {
        const workspaceFolders = vscode.workspace.workspaceFolders;
        const workspaceRoot = (workspaceFolders && workspaceFolders.length > 0) ? workspaceFolders[0].uri.fsPath : null;

        const candidates = [];
        if (workspaceRoot) {
            candidates.push(path.resolve(workspaceRoot, modelPath));
            candidates.push(path.resolve(workspaceRoot, 'checkpoints', modelPath));
            candidates.push(path.resolve(workspaceRoot, 'checkpoints', 'baseline', path.basename(modelPath)));
            candidates.push(path.resolve(workspaceRoot, 'checkpoints', path.basename(modelPath)));
        }
        candidates.push(path.resolve(__dirname, modelPath));
        candidates.push(path.resolve(__dirname, 'checkpoints', modelPath));
        candidates.push(path.resolve(__dirname, 'checkpoints', 'baseline', path.basename(modelPath)));
        candidates.push(path.resolve(__dirname, 'checkpoints', path.basename(modelPath)));

        const found = candidates.find(c => fs.existsSync(c));
        if (found) {
            resolvedModelPath = found;
        } else if (workspaceRoot) {
            resolvedModelPath = path.resolve(workspaceRoot, modelPath);
        } else {
            resolvedModelPath = path.resolve(__dirname, modelPath);
        }
    }

    log(`Spawning daemon: "${pythonPath}" "${serverScript}" --checkpoint "${resolvedModelPath}"`);

    try {
        pythonProcess = spawn(pythonPath, [serverScript, '--checkpoint', resolvedModelPath], {
            cwd: __dirname
        });
    } catch (e) {
        log(`Failed to spawn Python process: ${e.message}`);
        vscode.window.showErrorMessage(`APL IntelliSense SLM: Failed to launch Python interpreter "${pythonPath}". Please check the pythonPath setting.`);
        return;
    }

    let stdoutBuffer = '';
    pythonProcess.stdout.on('data', (data) => {
        stdoutBuffer += data.toString();
        let newlineIndex;
        while ((newlineIndex = stdoutBuffer.indexOf('\n')) !== -1) {
            const line = stdoutBuffer.substring(0, newlineIndex).trim();
            stdoutBuffer = stdoutBuffer.substring(newlineIndex + 1);

            if (!line) continue;

            try {
                const response = JSON.parse(line);
                const id = response.id;
                const resolve = activeRequests.get(id);
                if (resolve) {
                    resolve(response.completion);
                    activeRequests.delete(id);
                }
            } catch (e) {
                log(`Failed to parse daemon response: "${line}". Error: ${e.message}`);
            }
        }
    });

    pythonProcess.stderr.on('data', (data) => {
        log(`Daemon stderr: ${data.toString().trim()}`);
    });

    pythonProcess.on('error', (err) => {
        log(`Daemon process error: ${err.message}`);
        vscode.window.showErrorMessage(`APL IntelliSense SLM: Daemon process error: ${err.message}`);
    });

    pythonProcess.on('close', (code) => {
        log(`Daemon process exited with code ${code}`);
        pythonProcess = null;
    });
}

function stopDaemon() {
    if (pythonProcess) {
        log('Stopping daemon process...');
        pythonProcess.kill();
        pythonProcess = null;
    }
    activeRequests.clear();
}

function queryDaemon(prompt) {
    return new Promise((resolve) => {
        if (!pythonProcess) {
            startDaemon();
        }
        if (!pythonProcess) {
            resolve('');
            return;
        }

        const id = ++requestCounter;
        activeRequests.set(id, resolve);

        const config = vscode.workspace.getConfiguration('apl-slm');
        const legacyConfig = vscode.workspace.getConfiguration('apl-llm');
        const maxTokens = config.get('maxTokens') || legacyConfig.get('maxTokens') || 128;

        const request = JSON.stringify({ id: id, prompt: prompt, max_tokens: maxTokens }) + '\n';
        
        try {
            pythonProcess.stdin.write(request);
        } catch (e) {
            log(`Failed to write to daemon: ${e.message}`);
            activeRequests.delete(id);
            resolve('');
            return;
        }

        const timeoutMs = Math.max(4000, maxTokens * 15);
        setTimeout(() => {
            if (activeRequests.has(id)) {
                log(`Request ${id} timed out.`);
                activeRequests.delete(id);
                resolve('');
            }
        }, timeoutMs);
    });
}

function activate(context) {
    outputChannel = vscode.window.createOutputChannel("APL IntelliSense SLM");
    context.subscriptions.push(outputChannel);
    log("APL IntelliSense SLM Extension Activated!");

    startDaemon();

    context.subscriptions.push(
        vscode.workspace.onDidChangeConfiguration(e => {
            if (e.affectsConfiguration('apl-slm.pythonPath') || e.affectsConfiguration('apl-slm.modelPath') ||
                e.affectsConfiguration('apl-llm.pythonPath') || e.affectsConfiguration('apl-llm.modelPath')) {
                log("Configuration changed. Restarting daemon...");
                stopDaemon();
                startDaemon();
            }
        })
    );

    // Register Inline Completion Provider
    const provider = {
        async provideInlineCompletionItems(document, position, context, token) {
            const lineText = document.lineAt(position.line).text;
            const prompt = lineText.substring(0, position.character);

            if (!prompt.trim()) {
                return [];
            }

            if (token.isCancellationRequested) {
                return [];
            }

            try {
                const completion = await queryDaemon(prompt);
                if (token.isCancellationRequested || !completion) {
                    return [];
                }

                const item = new vscode.InlineCompletionItem(completion);
                item.range = new vscode.Range(position, position);
                return [item];
            } catch (e) {
                log(`Completion provider error: ${e.message}`);
                return [];
            }
        }
    };

    context.subscriptions.push(
        vscode.languages.registerInlineCompletionItemProvider(
            { pattern: '**/*.{apl,apln,aplf,aplo,aplc,dyalog}' },
            provider
        )
    );

    // Command to restart Daemon
    context.subscriptions.push(
        vscode.commands.registerCommand('apl-slm.restartDaemon', () => {
            stopDaemon();
            startDaemon();
            vscode.window.showInformationMessage('APL IntelliSense SLM daemon restarted.');
        })
    );
    context.subscriptions.push(
        vscode.commands.registerCommand('apl-llm.restartDaemon', () => {
            stopDaemon();
            startDaemon();
            vscode.window.showInformationMessage('APL IntelliSense SLM daemon restarted.');
        })
    );
}

function deactivate() {
    stopDaemon();
}

module.exports = {
    activate,
    deactivate
};
