//! Shared backend discovery and device descriptions for native inference.

use serde::Serialize;

use crate::DevicePreference;

pub const BACKEND_REGISTRY_VERSION: u32 = 1;
pub const GPU_MEMORY_SAFETY_MARGIN: u64 = 512 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InferencePolicy {
    pub preference: DevicePreference,
    pub selector: String,
    pub explicit: bool,
    pub cuda_device_index: Option<i32>,
}

impl InferencePolicy {
    pub fn from_selector(selector: Option<&str>, no_gpu: bool) -> crate::Result<Self> {
        if no_gpu {
            return Ok(Self {
                preference: DevicePreference::Cpu,
                selector: "cpu".into(),
                explicit: true,
                cuda_device_index: None,
            });
        }
        let selector = selector
            .map(str::trim)
            .filter(|selector| !selector.is_empty())
            .unwrap_or("auto")
            .to_ascii_lowercase();
        let preference = DevicePreference::parse(&selector)?;
        let cuda_device_index = selector
            .strip_prefix("cuda:")
            .map(|index| {
                index.parse::<i32>().map_err(|_| {
                    crate::Error::InvalidGguf(format!(
                        "unsupported CUDA device selector `{selector}`; expected cuda:INDEX"
                    ))
                })
            })
            .transpose()?;
        if cuda_device_index.is_some_and(|index| index < 0) {
            return Err(crate::Error::InvalidGguf(format!(
                "unsupported CUDA device selector `{selector}`; index must be non-negative"
            )));
        }
        let explicit = selector != "auto";
        Ok(Self {
            preference,
            selector,
            explicit,
            cuda_device_index,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendKind {
    Cpu,
    Metal,
    Cuda,
}

impl BackendKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceType {
    Cpu,
    IntegratedGpu,
    DiscreteGpu,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeviceInfo {
    pub backend: BackendKind,
    pub id: String,
    pub name: String,
    pub description: String,
    pub device_type: DeviceType,
    pub memory_free: Option<u64>,
    pub memory_total: Option<u64>,
    pub compute_capability: Option<String>,
    pub metal_family: Option<String>,
    pub capabilities: Vec<String>,
    pub rejection_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BackendProbe {
    pub backend: BackendKind,
    pub backend_id: String,
    pub build_info: String,
    pub abi_version: u32,
    pub compiled: bool,
    pub available: bool,
    pub score: i32,
    pub reason: Option<String>,
    pub devices: Vec<DeviceInfo>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InferenceBackendRegistry {
    pub version: u32,
    pub preference: String,
    pub explicit: bool,
    pub required_gpu_memory: u64,
    pub selected_backend: Option<BackendKind>,
    pub selected_device_id: Option<String>,
    pub probes: Vec<BackendProbe>,
}

impl InferenceBackendRegistry {
    pub fn probe(preference: DevicePreference, explicit: bool) -> Self {
        Self::probe_with_required_memory(preference, explicit, 0)
    }

    pub fn probe_with_required_memory(
        preference: DevicePreference,
        explicit: bool,
        required_gpu_memory: u64,
    ) -> Self {
        let mut probes = vec![probe_cuda(), probe_metal(), probe_cpu()];
        apply_memory_requirement(&mut probes, required_gpu_memory);
        let selected_backend = select_backend(&probes, &preference);
        let selected_device_id =
            selected_device(&probes, selected_backend).map(|device| device.id.clone());
        Self {
            version: BACKEND_REGISTRY_VERSION,
            preference: preference.as_str().to_string(),
            explicit,
            required_gpu_memory,
            selected_backend,
            selected_device_id,
            probes,
        }
    }

    pub fn probe_policy(policy: &InferencePolicy, required_gpu_memory: u64) -> Self {
        let mut registry = Self::probe_with_required_memory(
            policy.preference.clone(),
            policy.explicit,
            required_gpu_memory,
        );
        registry.preference.clone_from(&policy.selector);
        if let Some(index) = policy.cuda_device_index {
            select_explicit_cuda_device(&mut registry, index);
        }
        registry
    }

    pub fn selected_probe(&self) -> Option<&BackendProbe> {
        let selected = self.selected_backend?;
        self.probes.iter().find(|probe| probe.backend == selected)
    }

    pub fn is_satisfied(&self) -> bool {
        self.selected_probe().is_some_and(|probe| probe.available)
    }
}

fn select_explicit_cuda_device(registry: &mut InferenceBackendRegistry, index: i32) {
    let requested = format!("cuda:{index}");
    let Some(probe) = registry
        .probes
        .iter_mut()
        .find(|probe| probe.backend == BackendKind::Cuda)
    else {
        registry.selected_backend = None;
        registry.selected_device_id = None;
        return;
    };
    for device in &mut probe.devices {
        if device.id != requested && device.rejection_reason.is_none() {
            device.rejection_reason = Some(format!("device {requested} was explicitly requested"));
        }
    }
    let selected_device_id = probe
        .devices
        .iter()
        .find(|device| device.id == requested && device.rejection_reason.is_none())
        .map(|device| device.id.clone());
    probe.available = selected_device_id.is_some();
    if selected_device_id.is_none() {
        probe.score = 0;
        probe.reason = Some(format!(
            "explicit CUDA device {requested} is absent, incompatible, or lacks free memory"
        ));
    }
    registry.selected_backend = selected_device_id.as_ref().map(|_| BackendKind::Cuda);
    registry.selected_device_id = selected_device_id;
}

fn apply_memory_requirement(probes: &mut [BackendProbe], required: u64) {
    if required == 0 {
        return;
    }
    for probe in probes.iter_mut().filter(|probe| {
        matches!(probe.backend, BackendKind::Cuda | BackendKind::Metal) && probe.compiled
    }) {
        for device in &mut probe.devices {
            if device.rejection_reason.is_none() && !device_has_memory(device, required) {
                device.rejection_reason = Some(format!(
                    "{} bytes free, {} bytes plus {} byte safety margin required",
                    device.memory_free.unwrap_or(0),
                    required,
                    GPU_MEMORY_SAFETY_MARGIN
                ));
            }
        }
        probe.available = probe
            .devices
            .iter()
            .any(|device| device.rejection_reason.is_none());
        if !probe.available {
            probe.score = 0;
            probe.reason = Some("no compatible device has enough free memory".into());
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceModelKind {
    EmbeddingGemma,
    Qwen35,
}

pub fn estimated_gpu_memory(model: InferenceModelKind, model_file_bytes: u64) -> u64 {
    let workspace = match model {
        InferenceModelKind::EmbeddingGemma => 768 * 1024 * 1024,
        InferenceModelKind::Qwen35 => 1024 * 1024 * 1024,
    };
    model_file_bytes.saturating_add(workspace)
}

pub fn device_has_memory(device: &DeviceInfo, required: u64) -> bool {
    device
        .memory_free
        .is_none_or(|free| free >= required.saturating_add(GPU_MEMORY_SAFETY_MARGIN))
}

pub fn preflight_explicit_model(
    policy: &InferencePolicy,
    model: InferenceModelKind,
    model_file_bytes: u64,
) -> crate::Result<()> {
    if !policy.explicit
        || !matches!(
            policy.preference,
            DevicePreference::Metal | DevicePreference::Cuda
        )
    {
        return Ok(());
    }
    let required = estimated_gpu_memory(model, model_file_bytes);
    let registry = InferenceBackendRegistry::probe_policy(policy, required);
    if registry.is_satisfied() {
        return Ok(());
    }
    let requested = match policy.preference {
        DevicePreference::Metal => BackendKind::Metal,
        DevicePreference::Cuda => BackendKind::Cuda,
        DevicePreference::Auto | DevicePreference::Cpu => unreachable!(),
    };
    let reason = registry
        .probes
        .iter()
        .find(|probe| probe.backend == requested)
        .and_then(|probe| probe.reason.as_deref())
        .unwrap_or("backend probe rejected the requested device");
    Err(crate::Error::InvalidGguf(format!(
        "explicit inference backend `{}` is unavailable: {reason}",
        policy.selector
    )))
}

fn select_backend(probes: &[BackendProbe], preference: &DevicePreference) -> Option<BackendKind> {
    match preference {
        DevicePreference::Cpu => available(probes, BackendKind::Cpu),
        DevicePreference::Metal => available(probes, BackendKind::Metal),
        DevicePreference::Cuda => available(probes, BackendKind::Cuda),
        DevicePreference::Auto => probes
            .iter()
            .filter(|probe| probe.available)
            .max_by_key(|probe| probe.score)
            .map(|probe| probe.backend),
    }
}

fn available(probes: &[BackendProbe], backend: BackendKind) -> Option<BackendKind> {
    probes
        .iter()
        .any(|probe| probe.backend == backend && probe.available)
        .then_some(backend)
}

fn selected_device(probes: &[BackendProbe], backend: Option<BackendKind>) -> Option<&DeviceInfo> {
    let backend = backend?;
    probes
        .iter()
        .find(|probe| probe.backend == backend && probe.available)?
        .devices
        .iter()
        .find(|device| device.rejection_reason.is_none())
}

fn probe_cpu() -> BackendProbe {
    let capabilities = cpu_capabilities();
    let (memory_free, memory_total) = system_memory();
    BackendProbe {
        backend: BackendKind::Cpu,
        backend_id: "greppy-cpu-q4k-v1".into(),
        build_info: format!("rust-{}-{}", std::env::consts::ARCH, std::env::consts::OS),
        abi_version: BACKEND_REGISTRY_VERSION,
        compiled: true,
        available: true,
        score: 100,
        reason: None,
        devices: vec![DeviceInfo {
            backend: BackendKind::Cpu,
            id: "cpu:0".into(),
            name: std::env::consts::ARCH.into(),
            description: format!("{} CPU", std::env::consts::ARCH),
            device_type: DeviceType::Cpu,
            memory_free,
            memory_total,
            compute_capability: None,
            metal_family: None,
            capabilities,
            rejection_reason: None,
        }],
    }
}

fn cpu_capabilities() -> Vec<String> {
    let mut out = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        let features = crate::cpu_features::detected();
        for (name, enabled) in [
            ("sse4.2", features.sse42),
            ("avx2", features.avx2),
            ("fma", features.fma),
            ("avx-vnni", features.avx_vnni),
            ("avx512f", features.avx512f),
        ] {
            if enabled {
                out.push(name.to_string());
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        let features = crate::cpu_features::detected();
        out.push("neon".into());
        if features.dotprod {
            out.push("dotprod".into());
        }
        if features.i8mm {
            out.push("i8mm".into());
        }
    }
    out
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn system_memory() -> (Option<u64>, Option<u64>) {
    let mut info = std::mem::MaybeUninit::<libc::sysinfo>::zeroed();
    if unsafe { libc::sysinfo(info.as_mut_ptr()) } != 0 {
        return (None, None);
    }
    let info = unsafe { info.assume_init() };
    let unit = u64::from(info.mem_unit);
    (
        Some(info.freeram.saturating_mul(unit)),
        Some(info.totalram.saturating_mul(unit)),
    )
}

#[cfg(target_os = "macos")]
fn system_memory() -> (Option<u64>, Option<u64>) {
    let mut total = 0u64;
    let mut size = std::mem::size_of_val(&total);
    let result = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&mut total as *mut u64).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    (None, (result == 0).then_some(total))
}

#[cfg(not(any(target_os = "linux", target_os = "android", target_os = "macos")))]
fn system_memory() -> (Option<u64>, Option<u64>) {
    (None, None)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn probe_metal() -> BackendProbe {
    match crate::metal::ffi::global_device() {
        Some(device) if device.runtime_ready() => BackendProbe {
            backend: BackendKind::Metal,
            backend_id: "greppy-metal-q4k-v1".into(),
            build_info: device.build_info(),
            abi_version: BACKEND_REGISTRY_VERSION,
            compiled: true,
            available: true,
            score: 300,
            reason: None,
            devices: vec![device.device_info()],
        },
        Some(_) => unavailable_probe(
            BackendKind::Metal,
            true,
            "embedded Metal library is unavailable",
        ),
        None => unavailable_probe(BackendKind::Metal, true, "no Metal device found"),
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn probe_metal() -> BackendProbe {
    unavailable_probe(
        BackendKind::Metal,
        false,
        "Metal backend is not compiled for this platform",
    )
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
fn probe_cuda() -> BackendProbe {
    match crate::cuda::ffi::probe_devices() {
        Ok(devices) if !devices.is_empty() => {
            let available = devices
                .iter()
                .any(|device| device.rejection_reason.is_none());
            BackendProbe {
                backend: BackendKind::Cuda,
                backend_id: "greppy-cuda-q4k-v1".into(),
                build_info: crate::cuda::ffi::backend_build_info(),
                abi_version: crate::cuda::ffi::backend_abi_version().unwrap_or(0),
                compiled: true,
                available,
                score: if available { 400 } else { 0 },
                reason: (!available).then_some("no compatible CUDA device found".into()),
                devices,
            }
        }
        Ok(_) => unavailable_probe(BackendKind::Cuda, true, "no compatible CUDA device found"),
        Err(error) => unavailable_probe(BackendKind::Cuda, true, &error.to_string()),
    }
}

#[cfg(not(all(feature = "cuda", any(target_os = "linux", target_os = "windows"))))]
fn probe_cuda() -> BackendProbe {
    unavailable_probe(
        BackendKind::Cuda,
        false,
        "CUDA backend is not compiled for this platform",
    )
}

fn unavailable_probe(backend: BackendKind, compiled: bool, reason: &str) -> BackendProbe {
    BackendProbe {
        backend,
        backend_id: format!("greppy-{}-q4k-v1", backend.as_str()),
        build_info: format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
        abi_version: BACKEND_REGISTRY_VERSION,
        compiled,
        available: false,
        score: 0,
        reason: Some(reason.to_string()),
        devices: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_is_always_a_valid_auto_fallback() {
        let registry = InferenceBackendRegistry::probe(DevicePreference::Auto, false);
        assert!(registry.probes.iter().any(|probe| {
            probe.backend == BackendKind::Cpu && probe.compiled && probe.available
        }));
        assert!(registry.is_satisfied());
    }

    #[test]
    fn explicit_uncompiled_backend_is_not_satisfied() {
        let probes = vec![
            probe_cpu(),
            unavailable_probe(BackendKind::Cuda, false, "test"),
        ];
        assert_eq!(select_backend(&probes, &DevicePreference::Cuda), None);
    }

    #[test]
    fn memory_requirement_rejects_gpu_before_selection() {
        let mut probe = BackendProbe {
            backend: BackendKind::Cuda,
            backend_id: "test".into(),
            build_info: "test".into(),
            abi_version: 1,
            compiled: true,
            available: true,
            score: 400,
            reason: None,
            devices: vec![DeviceInfo {
                backend: BackendKind::Cuda,
                id: "cuda:0".into(),
                name: "test".into(),
                description: "test".into(),
                device_type: DeviceType::DiscreteGpu,
                memory_free: Some(GPU_MEMORY_SAFETY_MARGIN),
                memory_total: Some(GPU_MEMORY_SAFETY_MARGIN),
                compute_capability: Some("8.6".into()),
                metal_family: None,
                capabilities: Vec::new(),
                rejection_reason: None,
            }],
        };
        apply_memory_requirement(std::slice::from_mut(&mut probe), 1);
        assert!(!probe.available);
        assert_eq!(probe.score, 0);
        assert!(probe.devices[0].rejection_reason.is_some());
    }

    #[test]
    fn policy_parses_explicit_cuda_index() {
        let policy = InferencePolicy::from_selector(Some("CUDA:2"), false).unwrap();
        assert_eq!(policy.preference, DevicePreference::Cuda);
        assert_eq!(policy.selector, "cuda:2");
        assert_eq!(policy.cuda_device_index, Some(2));
        assert!(policy.explicit);
        assert!(InferencePolicy::from_selector(Some("cuda:-1"), false).is_err());
    }

    #[test]
    fn explicit_cuda_index_selects_only_the_requested_device() {
        let mut registry = InferenceBackendRegistry {
            version: BACKEND_REGISTRY_VERSION,
            preference: "cuda:1".into(),
            explicit: true,
            required_gpu_memory: 0,
            selected_backend: Some(BackendKind::Cuda),
            selected_device_id: Some("cuda:0".into()),
            probes: vec![BackendProbe {
                backend: BackendKind::Cuda,
                backend_id: "test".into(),
                build_info: "test".into(),
                abi_version: 1,
                compiled: true,
                available: true,
                score: 400,
                reason: None,
                devices: vec![test_cuda_device("cuda:0"), test_cuda_device("cuda:1")],
            }],
        };
        select_explicit_cuda_device(&mut registry, 1);
        assert_eq!(registry.selected_backend, Some(BackendKind::Cuda));
        assert_eq!(registry.selected_device_id.as_deref(), Some("cuda:1"));
        assert!(registry.probes[0].devices[0].rejection_reason.is_some());
        assert!(registry.probes[0].devices[1].rejection_reason.is_none());

        select_explicit_cuda_device(&mut registry, 7);
        assert_eq!(registry.selected_backend, None);
        assert_eq!(registry.selected_device_id, None);
    }

    fn test_cuda_device(id: &str) -> DeviceInfo {
        DeviceInfo {
            backend: BackendKind::Cuda,
            id: id.into(),
            name: id.into(),
            description: "test".into(),
            device_type: DeviceType::DiscreteGpu,
            memory_free: Some(u64::MAX),
            memory_total: Some(u64::MAX),
            compute_capability: Some("8.6".into()),
            metal_family: None,
            capabilities: Vec::new(),
            rejection_reason: None,
        }
    }

    #[test]
    fn gpu_memory_estimate_includes_workspace_and_margin_is_separate() {
        let model = 512 * 1024 * 1024;
        let required = estimated_gpu_memory(InferenceModelKind::Qwen35, model);
        assert!(required > model);
        let device = DeviceInfo {
            backend: BackendKind::Cuda,
            id: "cuda:0".into(),
            name: "test".into(),
            description: "test".into(),
            device_type: DeviceType::DiscreteGpu,
            memory_free: Some(required + GPU_MEMORY_SAFETY_MARGIN),
            memory_total: None,
            compute_capability: Some("8.6".into()),
            metal_family: None,
            capabilities: Vec::new(),
            rejection_reason: None,
        };
        assert!(device_has_memory(&device, required));
    }

    #[test]
    fn cpu_and_auto_preflight_do_not_probe_gpu_backends() {
        let cpu = InferencePolicy::from_selector(Some("cpu"), false).unwrap();
        preflight_explicit_model(&cpu, InferenceModelKind::Qwen35, u64::MAX).unwrap();
        let auto = InferencePolicy::from_selector(None, false).unwrap();
        preflight_explicit_model(&auto, InferenceModelKind::Qwen35, u64::MAX).unwrap();
    }
}
