import AppKit
import Sparkle

// Held at module scope so ARC does not release the updater before the run loop exits.
private var activeUpdater: SPUUpdater?

private final class AppDelegate: NSObject, NSApplicationDelegate, SPUUpdaterDelegate {
    private let isBackgroundCheck: Bool

    init(backgroundCheck: Bool) {
        self.isBackgroundCheck = backgroundCheck
        super.init()
    }

    func applicationDidFinishLaunching(_: Notification) {
        // Never appear in Dock or steal focus on launch.
        NSApp.setActivationPolicy(.prohibited)

        let hostBundle = resolveHostBundle()
        let userDriver = SPUStandardUserDriver(hostBundle: hostBundle, delegate: nil)

        do {
            let updater = try SPUUpdater(
                hostBundle: hostBundle,
                applicationBundle: hostBundle,
                userDriver: userDriver,
                delegate: self
            )
            activeUpdater = updater
            try updater.start()

            if isBackgroundCheck {
                updater.checkForUpdatesInBackground()
                // Terminate after 45 s regardless, so we never leave a zombie process.
                DispatchQueue.main.asyncAfter(deadline: .now() + 45) { NSApp.terminate(nil) }
            } else {
                // Manual "Check for Updates" – show window, bring app to front.
                NSApp.setActivationPolicy(.accessory)
                NSApp.activate(ignoringOtherApps: true)
                updater.checkForUpdates()
            }
        } catch {
            NSApp.terminate(nil)
        }
    }

    // MARK: SPUUpdaterDelegate

    func updaterDidNotFindUpdate(_: SPUUpdater) {
        if isBackgroundCheck {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { NSApp.terminate(nil) }
        }
        // For manual checks Sparkle shows its own "up to date" sheet; leave the run loop running.
    }

    func updater(_: SPUUpdater, didAbortWithError _: Error) {
        if isBackgroundCheck {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { NSApp.terminate(nil) }
        }
    }

    func updater(_: SPUUpdater, willInstallUpdate _: SUAppcastItem) {
        // Sparkle will relaunch the app; nothing extra needed here.
    }

    // MARK: Private

    /// Resolves the host bundle from an env variable set by the Python launcher,
    /// with a path-based fallback (helper lives at Contents/MacOS/SparkleHelper).
    private func resolveHostBundle() -> Bundle {
        let envPath = ProcessInfo.processInfo.environment["FACE_LOCAL_BUNDLE_PATH"] ?? ""
        if !envPath.isEmpty, let bundle = Bundle(path: envPath) {
            return bundle
        }
        // Fallback: walk up three levels from the binary path.
        //   .../Face-Local.app/Contents/MacOS/SparkleHelper  ->  Face-Local.app
        let helperURL = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
        let appURL = helperURL
            .deletingLastPathComponent() // MacOS/
            .deletingLastPathComponent() // Contents/
            .deletingLastPathComponent() // Face-Local.app/
        return Bundle(url: appURL) ?? Bundle.main
    }
}

let isBackground = CommandLine.arguments.contains("--background")
let app = NSApplication.shared
let delegate = AppDelegate(backgroundCheck: isBackground)
app.delegate = delegate
app.run()
