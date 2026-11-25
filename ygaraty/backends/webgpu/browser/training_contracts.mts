// Shared contracts for compiled DiLoCo rounds and the HTTP training service.
import type {Tensor} from './sdk_tensor.mjs';
import type {DenseArithmetic} from './webgpu_contracts.mjs';
export type Precision = 'float64' | 'webgpu';
export type Named<T> = Record<string, T>;
export type Moments<T> = {m: Named<T>; v: Named<T>};
export const momentKinds = ['m', 'v'] as const;
export interface ArrayTensor {shape: number[]; data: number[]}
export type WireTensor = ArrayTensor |
  {shape: number[]; dtype: 'float32'; base64: string} |
  {shape: number[]; dtype: 'uint8'; normalize: 'x/127.5-1'; base64: string};
export interface Task {
  id: string; op: string; shape: number[];
  inputs: {value: string; region?: number[] | null}[];
  attrs: Record<string, unknown>;
}
export interface StepPlan {
  step: number; loss: string; tasks: Task[];
  constants: Named<WireTensor>; parameters: Named<string>; updates: Named<string>;
  state_parameters?: Moments<string>;
}
export interface Assignment {
  round_id: string; version: number; worker: string; steps: number;
  weights: Named<WireTensor>; wire_dtype?: 'float32';
}
export interface OptimizerState {step: number; moments: Moments<WireTensor> | null}
export interface ExecutionStats {cpu_tasks: number; gpu_tasks: number; matmul_tasks: number; optimizer_seconds?: number; resident_rounds?: number; gpu_submissions?: number; readbacks?: number; uploaded_bytes?: number; downloaded_bytes?: number; gpu_buffer_bytes?: number}
export interface RoundResult {
  stats: ExecutionStats;
  round_id: string; version: number; worker: string; steps: number;
  pseudo_gradient: Named<WireTensor>; losses: number[];
}
export interface AdapterInfo {
  vendor: string; architecture: string; device: string; description: string;
  isFallbackAdapter: boolean | null; maxBufferSize: number; maxStorageBufferBindingSize: number;
}
export interface ResidentTrainingRequest {
  steps: {plan: StepPlan; constants: Named<Tensor>}[];
  weights: Named<Tensor>; moments: Moments<Tensor>; signal?: AbortSignal;
}
export interface ResidentTrainingResult {
  weights: Named<Tensor>; moments: Moments<Tensor>; losses: number[]; stats: ExecutionStats;
}
export interface TrainingExecutor {
  executeTrainingRound?(request: ResidentTrainingRequest, arithmetic: DenseArithmetic): Promise<ResidentTrainingResult | null>;
  executeMuonBlock?(tasks: Task[], inputs: Map<string, Tensor>, outputs: string[],
    arithmetic: DenseArithmetic): Promise<Map<string, Tensor> | null>;
  execute(task: Task & {precision: Precision}, inputs: Tensor[]):
    Promise<{value: Tensor; backend: 'cpu' | 'webgpu'}>;
}
export interface ContributorSDK extends TrainingExecutor {
  device: GPUDevice | null; reason: string; adapterInfo: AdapterInfo | null;
  initialize(preferGPU?: boolean): Promise<string>;
}
export interface ExecutionOptions {precision?: Precision; arithmetic?: DenseArithmetic; signal?: AbortSignal}
export interface Credentials {worker: string; token: string; precision: Precision; heartbeat_seconds: number}
export interface SessionStatus {
  state: 'waiting' | 'training' | 'submitted' | 'complete' | 'failed';
  version?: number; rounds?: number; connected?: number; required?: number;
  pending?: string[]; retries?: number; error?: string | null; metrics?: unknown[];
}
export interface RoundBundle {
  state: 'assignment'; format: 'ygaraty-diloco-round-v1';
  optimizer: OptimizerState; assignment: Assignment; plans: StepPlan[];
}
export interface Submission {
  result: RoundResult; optimizer: OptimizerState;
  stats: ExecutionStats & {seconds: number; precision: Precision; adapter: AdapterInfo | null};
}
export type ContributorStatus = SessionStatus | {state: 'training'; version: number; worker: string} | {state: 'joined' | 'reconnecting' | 'stopped';
  worker?: string; backend?: string; message?: string};

export function object(value: unknown): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) throw Error('Expected object');
  return value as Record<string, unknown>;
}
const text = (value: unknown): string => {
  if (typeof value !== 'string') throw Error('Expected string');
  return value;
};
const integer = (value: unknown, minimum = 0): number => {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < minimum) throw Error('Expected integer');
  return value;
};
function array<T>(value: unknown, decode: (item: unknown) => T): T[] {
  if (!Array.isArray(value)) throw Error('Expected array');
  return value.map(item => decode(item));
}
function named<T>(value: unknown, decode: (item: unknown) => T): Named<T> {
  return Object.fromEntries(Object.entries(object(value)).map(([key, item]) => [key, decode(item)]));
}
export function decodeTensor(value: unknown): WireTensor {
  const raw = object(value), shape = array(raw.shape, n => integer(n, 1));
  if (shape.length !== 2) throw Error('Expected rank-two tensor');
  if (raw.dtype === 'float32') return {shape, dtype: 'float32', base64: text(raw.base64)};
  if (raw.dtype === 'uint8' && raw.normalize === 'x/127.5-1')
    return {shape, dtype: 'uint8', normalize: raw.normalize, base64: text(raw.base64)};
  if (raw.dtype !== undefined) throw Error('Unknown tensor encoding');
  const data = array(raw.data, x => {
    if (typeof x !== 'number' || !Number.isFinite(x)) throw Error('Nonfinite tensor');
    return x;
  });
  if (data.length !== shape[0] * shape[1]) throw Error('Invalid tensor size');
  return {shape, data};
}
function optimizer(value: unknown): OptimizerState {
  const raw = object(value), step = integer(raw.step);
  if (raw.moments === null && step === 0) return {step, moments: null};
  const moments = object(raw.moments);
  return {step, moments: {m: named(moments.m, decodeTensor), v: named(moments.v, decodeTensor)}};
}
function task(value: unknown): Task {
  const raw = object(value);
  return {id: text(raw.id), op: text(raw.op), shape: array(raw.shape, n => integer(n, 1)),
    attrs: object(raw.attrs), inputs: array(raw.inputs, item => {
      const input = object(item);
      return {value: text(input.value), region: input.region == null ? null : array(input.region, integer)};
    })};
}
export function decodeStatus(value: unknown): SessionStatus {
  const raw = object(value), state = raw.state;
  if (state !== 'waiting' && state !== 'training' && state !== 'submitted' && state !== 'complete' && state !== 'failed')
    throw Error('Invalid training session state');
  return {state, ...(raw.version === undefined ? {} : {version: integer(raw.version)}),
    ...(raw.rounds === undefined ? {} : {rounds: integer(raw.rounds)}),
    ...(raw.connected === undefined ? {} : {connected: integer(raw.connected)}),
    ...(raw.required === undefined ? {} : {required: integer(raw.required)}),
    ...(raw.retries === undefined ? {} : {retries: integer(raw.retries)}),
    ...(raw.pending === undefined ? {} : {pending: array(raw.pending, text)}),
    ...(raw.error == null ? {} : {error: text(raw.error)}),
    ...(raw.metrics === undefined ? {} : {metrics: array(raw.metrics, item => item)})};
}
export function decodeJoin(value: unknown): Credentials | SessionStatus {
  const raw = object(value);
  if ('state' in raw) return decodeStatus(raw);
  if ((raw.precision !== 'float64' && raw.precision !== 'webgpu') ||
      typeof raw.heartbeat_seconds !== 'number' || !Number.isFinite(raw.heartbeat_seconds) || raw.heartbeat_seconds <= 0)
    throw Error('Invalid contributor credentials');
  return {worker: text(raw.worker), token: text(raw.token), precision: raw.precision, heartbeat_seconds: raw.heartbeat_seconds};
}
export function decodeAssignment(value: unknown): RoundBundle | SessionStatus {
  const raw = object(value);
  if (raw.state !== 'assignment') return decodeStatus(raw);
  if (raw.format !== 'ygaraty-diloco-round-v1') throw Error('Unknown round format');
  const a = object(raw.assignment);
  if (a.wire_dtype !== undefined && a.wire_dtype !== 'float32') throw Error('Unknown wire dtype');
  return {state: 'assignment', format: raw.format, optimizer: optimizer(raw.optimizer),
    assignment: {round_id: text(a.round_id), version: integer(a.version), worker: text(a.worker),
      steps: integer(a.steps, 1), weights: named(a.weights, decodeTensor), wire_dtype: a.wire_dtype},
    plans: array(raw.plans, decodeStepPlan)};
}

export function decodeStepPlan(value: unknown): StepPlan {
  const p = object(value);
  return {step: integer(p.step, 1), loss: text(p.loss), tasks: array(p.tasks, task),
    constants: named(p.constants, decodeTensor), parameters: named(p.parameters, text), updates: named(p.updates, text),
    ...(p.state_parameters === undefined ? {} : {state_parameters: {
      m: named(object(p.state_parameters).m, text), v: named(object(p.state_parameters).v, text)}})};
}
