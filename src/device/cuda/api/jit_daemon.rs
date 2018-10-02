//! Parallel PTX compilation.
use device::cuda::api::wrapper::*;
use device::cuda::api::Module;
use errno::errno;
use ipc_channel::ipc;
use libc;
use prctl;
use std;
use std::ffi::CString;
use std::slice;

/// A process that compiles PTX in a separate process.
pub struct JITDaemon {
    daemon: libc::pid_t,
    ptx_sender: ipc::IpcBytesSender,
    cubin_receiver: ipc::IpcBytesReceiver,
}

impl JITDaemon {
    pub fn compile<'a>(&mut self, context: &'a CudaContext, code: &str) -> Module<'a> {
        debug!("compiling {}", code);
        unwrap!(self.ptx_sender.send(code.as_bytes()));
        let cubin = unwrap!(self.cubin_receiver.recv());
        Module::from_cubin(context, &cubin)
    }
}

impl Drop for JITDaemon {
    fn drop(&mut self) {
        unwrap!(self.ptx_sender.send(&[]));
        unsafe {
            if libc::waitpid(self.daemon, std::ptr::null_mut(), 0) == -1 {
                info!("unable to kill jit process: {}", errno());
            }
        }
    }
}

/// A function that listen for incomming PTX code and compiles it.
fn daemon(
    receiver: &ipc::IpcBytesReceiver,
    sender: &ipc::IpcBytesSender,
    opt_level: usize,
)
{
    unsafe {
        let ctx = init_cuda(0);
        loop {
            let code = receiver.recv().unwrap_or_else(|_| {
                error!("exiting PTX jit process because the receiver is broken");
                std::process::exit(0);
            });
            if code.is_empty() {
                trace!("exiting PTX jit process");
                return;
            }
            let code_len = code.len();
            let code = unwrap!(CString::new(code));
            let cubin = compile_ptx_to_cubin(ctx, code.as_ptr(), code_len, opt_level);
            unwrap!(sender.send(slice::from_raw_parts(cubin.data, cubin.data_size)));
            free_cubin_object(cubin);
        }
    }
}

/// Spawns daemons from a separate process, so a CUDA context can be created in
/// the main process without problems for the child processes.
pub struct DaemonSpawner {
    sender: ipc::IpcSender<Option<(ipc::IpcBytesReceiver, ipc::IpcBytesSender, usize)>>,
    receiver: ipc::IpcReceiver<libc::pid_t>,
    pid: libc::pid_t,
}

impl DaemonSpawner {
    /// Creates a new `DaemonSpawner`. The spawner must be called before any
    /// call to the cuda API.
    pub fn new() -> Self {
        // Set the current process as the manager of subprocesses. Otherwise, we can't
        // wait on grandchildrens.
        if prctl::set_child_subreaper(true).is_err() {
            panic!("unable to set the subreaper flag: {}", errno());
        }
        let (pid_sender, pid_receiver) = unwrap!(ipc::channel());
        let (channel_sender, channel_receiver) = unwrap!(ipc::channel());
        let pid = unsafe {
            fork_function(|| {
                loop {
                    let (receiver, sender, opt_level) = match channel_receiver.recv() {
                        Ok(Some(msg)) => msg,
                        Ok(None) => {
                            trace!("exiting PTX daemon spawner");
                            return;
                        }
                        Err(err) => {
                            error!("error in PTX daemon spawner: {:?}", err);
                            return;
                        }
                    };
                    // We daemonize the JIT so it is attached to the closet subreaper
                    // process, i.e. the main process.
                    let tmp_pid = fork_function(|| {
                        let pid = fork_function(|| daemon(&receiver, &sender, opt_level));
                        if let Err(err) = pid_sender.send(pid) {
                            error!(
                                "error in PTX daemon spawner (sending PID): {:?}",
                                err
                            );
                            return;
                        }
                    });
                    if libc::waitpid(tmp_pid, std::ptr::null_mut(), 0) == -1 {
                        panic!("unable to kill daemonizing process: {}", errno());
                    }
                }
            })
        };
        DaemonSpawner {
            sender: channel_sender,
            receiver: pid_receiver,
            pid,
        }
    }

    /// Creates a new `JITDaemon`.
    pub fn spawn_jit(&self, opt_level: usize) -> JITDaemon {
        let (ptx_sender, ptx_receiver) = unwrap!(ipc::bytes_channel());
        let (cubin_sender, cubin_receiver) = unwrap!(ipc::bytes_channel());
        unwrap!(
            self.sender
                .send(Some((ptx_receiver, cubin_sender, opt_level)))
        );
        let daemon = unwrap!(self.receiver.recv());
        JITDaemon {
            daemon,
            ptx_sender,
            cubin_receiver,
        }
    }
}

impl Drop for DaemonSpawner {
    fn drop(&mut self) {
        unwrap!(self.sender.send(None));
        unsafe {
            if libc::waitpid(self.pid, std::ptr::null_mut(), 0) == -1 {
                panic!("unable to kill jit spawner process: {}", errno());
            }
        }
    }
}

/// Spawns a function in a new process. This function should not access shared
/// data structures that may have been touched by other threads at the time of
/// the fork.
unsafe fn fork_function<F: FnOnce()>(f: F) -> libc::pid_t {
    match libc::fork() {
        -1 => panic!("could no forl the process: {}", errno()),
        0 => {
            f();
            libc::exit(0)
        }
        pid => pid,
    }
}
