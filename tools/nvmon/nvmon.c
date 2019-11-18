/*
 * Helper script to monitor nvidia GPUs performance.
 *
 * It will print to stdout events of the following kinds:
 *
 *  - name, <gpu name>
 *  	occurs once and prints the GPU name.
 *
 *  - applicationClocks, <mClock in MHz>, <pClock in MHz>
 *  	occurs once and prints the current memory and GPU application clocks
 *
 *  - start, <time in microseconds>
 *  	occurs once and prints the start time.  There might be event (typically
 *  	pClock and mClock events) going further back than the start time if
 *  	they were already present in the buffer.
 *
 *  - temperature, <time in microseconds>, <temperature in C>
 *  	when the temperature goes above or beyond 90% of the GPU slowdown
 * threshold
 *
 *  - thermalHigh, <start time in microseconds>, <end time in microsecond>
 *  	with the interval of time where the GPU was above 90% of the slowdown
 * threshold
 *
 *  - thermalViolation, <range start in microseconds>, <range end in
 * microseconds>, 1 with the interval of time during which a thermal violation
 * occured (the thermal violation occured at some point during the interval, not
 * the whole interval)
 *
 *  - powerViolation, <range start in microseconds>, <range end in
 * microseconds>, <duration in nanoseconds> with the interval of time during
 * which a power violation occured (as in thermal violation it occured at some
 * point during the interval not the whole interval).  The duration in
 * nanoseconds *should* be the actual duration of the violation but this doesn't
 * match observations, so I am not sure what it measures exactly.  This is the
 * difference in violationTime from nvmlDeviceGetViolationStatus.
 *
 *  - pClock, <time in microseconds>, <pClock in MHz>
 *  	with the time at which the GPU clock changed and the new value
 *
 *  - mClock, <time in microseconds>, <pClock in MHz>
 *  	with the time at which the RAM clock changed and the new value
 *
 * This runs as a daemon and polls values every second.  It needs to be killed
 * using ^C.  Note that even though we run every second we get processor and
 * memory samples with a somewhat finer granularity (at least according to the
 * documentation -- nvmlDeviceGetSamples don't seem to update at more than
 * 1/sec anyways).
 */

#include <assert.h>
#include <nvml.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define NVML_CHECK(f)                                                          \
  {                                                                            \
    nvmlReturn_t __status = (f);                                               \
    if (__status != NVML_SUCCESS) {                                            \
      fprintf(stderr, "%s:%d:FAILURE: %d", __FILE__, __LINE__, __status);      \
      exit(1);                                                                 \
    }                                                                          \
  }

struct samplesBuffer {
  nvmlSamplingType_t type;
  nvmlValueType_t sampleValType;
  nvmlSample_t lastValue;
  unsigned long long lastSeen;

  nvmlSample_t *samples;
  unsigned int length;
  unsigned int capacity;
};

typedef struct samplesBuffer *samplesBuffer_t;

void samplesBufferUpdateViolationStatus(samplesBuffer_t buffer,
                                        const char *name, unsigned int tol) {
  assert(buffer != NULL);

  switch (buffer->sampleValType) {
  case NVML_VALUE_TYPE_UNSIGNED_INT:

    break;

  default:
    fprintf(stderr, "Invalid buffer sample type\n");
    exit(1);
  }

  if (buffer->length < 1) {
    return;
  }

  unsigned int pre = buffer->lastValue.sampleValue.uiVal;
  for (unsigned int i = 1; i < buffer->length; ++i) {
    unsigned int curr = buffer->samples[i].sampleValue.uiVal;

    if ((curr > pre && curr - pre > tol) || (curr < pre && pre - curr > tol)) {
      printf("%s, %llu, %u\n", name, buffer->samples[i].timeStamp, curr);

      pre = curr;
      buffer->lastValue = buffer->samples[i];
    }
  }

  buffer->lastSeen = buffer->lastValue.timeStamp;
}

void samplesBufferResizeForDevice(samplesBuffer_t buffer, nvmlDevice_t device) {
  assert(buffer != NULL);

  nvmlValueType_t sampleValType;
  unsigned int capacity;
  NVML_CHECK(nvmlDeviceGetSamples(device, buffer->type, 0, &sampleValType,
                                  &capacity, NULL));

  nvmlSample_t *samples = malloc(capacity * sizeof(nvmlSample_t));
  if (samples == NULL && capacity > 0) {
    return;
  }

  if (buffer->samples != NULL) {
    free(buffer->samples);
  }

  buffer->length = 0;
  buffer->samples = samples;
  buffer->capacity = capacity;
}

void samplesBufferInitForDevice(samplesBuffer_t buffer, nvmlDevice_t device,
                                nvmlSamplingType_t type) {
  assert(buffer != NULL);

  buffer->type = type;
  buffer->lastSeen = 0;
  buffer->samples = NULL;
  buffer->length = 0;
  buffer->capacity = 0;
  samplesBufferResizeForDevice(buffer, device);
}

void samplesBufferDestroy(samplesBuffer_t buffer) {
  assert(buffer != NULL);

  if (buffer->samples != NULL) {
    free(buffer->samples);
  }

  buffer->samples = NULL;
  buffer->length = 0;
  buffer->capacity = 0;
}

void samplesBufferFillFromDevice(samplesBuffer_t buffer, nvmlDevice_t device) {
  nvmlReturn_t status;

  buffer->length = buffer->capacity;
  status = nvmlDeviceGetSamples(device, buffer->type, buffer->lastSeen,
                                &buffer->sampleValType, &buffer->length,
                                buffer->samples);

  if (status == NVML_ERROR_NOT_FOUND) {
    buffer->length = 0;
  } else {
    NVML_CHECK(status);

    if (buffer->lastSeen == 0 && buffer->length > 0) {
      buffer->lastValue = buffer->samples[0];
      buffer->lastSeen = buffer->lastValue.timeStamp;
    }
  }
}

struct deviceInfo {
  unsigned int temperatureThresholdShutdown;
  unsigned int temperatureThresholdSlowdown;

  unsigned long long thermalHighReferenceTime;
  unsigned int thermalHigh;

  nvmlViolationTime_t powerViolationTime;
  // violationTime is 0/1 boolean
  nvmlViolationTime_t thermalViolationTime;
};

typedef struct deviceInfo *deviceInfo_t;

void deviceInfoUpdateViolationStatus(deviceInfo_t info, nvmlDevice_t device) {
  nvmlReturn_t status;
  nvmlViolationTime_t violationTime;

  unsigned int temperature;
  NVML_CHECK(
      nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature));

  status = nvmlDeviceGetViolationStatus(device, NVML_PERF_POLICY_THERMAL,
                                        &violationTime);

  if (status == NVML_ERROR_NOT_SUPPORTED) {
    violationTime.violationTime = 0;
    violationTime.referenceTime = 0;
  } else {
    NVML_CHECK(status);
  }

  unsigned long long thermalDelta =
      violationTime.violationTime - info->thermalViolationTime.violationTime;
  unsigned long long thermalStartTime =
      info->thermalViolationTime.referenceTime;
  info->thermalViolationTime = violationTime;

  NVML_CHECK(nvmlDeviceGetViolationStatus(device, NVML_PERF_POLICY_POWER,
                                          &violationTime));
  unsigned long long powerDelta =
      violationTime.violationTime - info->powerViolationTime.violationTime;
  unsigned long long powerStartTime = info->powerViolationTime.referenceTime;
  info->powerViolationTime = violationTime;

  if (info->thermalHigh !=
      (temperature >
       (unsigned int)(0.9 * info->temperatureThresholdSlowdown))) {
    printf("temperature, %llu, %u\n", info->thermalViolationTime.referenceTime,
           temperature);

    if (info->thermalHigh) {
      // End of thermal high
      printf("thermalHigh, %llu, %llu\n", info->thermalHighReferenceTime,
             info->thermalViolationTime.referenceTime);

      info->thermalHigh = 0;
    } else {
      // Start of thermal high
      info->thermalHigh = 1;
      info->thermalHighReferenceTime = info->thermalViolationTime.referenceTime;
    }
  }

  if (thermalDelta != 0) {
    printf("thermalViolation, %llu, %llu, %llu\n", thermalStartTime,
           info->thermalViolationTime.referenceTime, thermalDelta);
  }

  if (powerDelta != 0) {
    printf("powerViolation, %llu, %llu, %llu\n", powerStartTime,
           info->powerViolationTime.referenceTime, powerDelta);
  }
}

void deviceInfoInitForDevice(deviceInfo_t info, nvmlDevice_t device) {
  assert(info != NULL);

  info->thermalHighReferenceTime = 0;
  info->thermalHigh = 0;

  NVML_CHECK(nvmlDeviceGetTemperatureThreshold(
      device, NVML_TEMPERATURE_THRESHOLD_SHUTDOWN,
      &info->temperatureThresholdShutdown));
  NVML_CHECK(nvmlDeviceGetTemperatureThreshold(
      device, NVML_TEMPERATURE_THRESHOLD_SLOWDOWN,
      &info->temperatureThresholdSlowdown));

  nvmlReturn_t status = nvmlDeviceGetViolationStatus(
      device, NVML_PERF_POLICY_THERMAL, &info->thermalViolationTime);

  if (status == NVML_ERROR_NOT_SUPPORTED) {
    info->thermalViolationTime.violationTime = 0;
    info->thermalViolationTime.referenceTime = 0;
  } else {
    NVML_CHECK(status);
  }

  NVML_CHECK(nvmlDeviceGetViolationStatus(device, NVML_PERF_POLICY_POWER,
                                          &info->powerViolationTime));

  printf("start, %llu\n", info->powerViolationTime.referenceTime);
}

int main() {
  NVML_CHECK(nvmlInit());

  nvmlDevice_t device;
  NVML_CHECK(nvmlDeviceGetHandleByIndex(0, &device));

  char name[NVML_DEVICE_NAME_BUFFER_SIZE];
  NVML_CHECK(nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE));
  printf("name, %s\n", name);

  unsigned int mClock, pClock;
  NVML_CHECK(nvmlDeviceGetApplicationsClock(device, NVML_CLOCK_MEM, &mClock));
  NVML_CHECK(nvmlDeviceGetApplicationsClock(device, NVML_CLOCK_SM, &pClock));
  printf("applicationClocks, %u, %u\n", mClock, pClock);

  nvmlEnableState_t persistenceMode;
  NVML_CHECK(nvmlDeviceGetPersistenceMode(device, &persistenceMode));

  if (persistenceMode != NVML_FEATURE_ENABLED) {
    printf("Persistence mode is disabled.");
    exit(1);
  }

  struct samplesBuffer pClockSamples;
  samplesBufferInitForDevice(&pClockSamples, device,
                             NVML_PROCESSOR_CLK_SAMPLES);
  struct samplesBuffer mClockSamples;
  samplesBufferInitForDevice(&mClockSamples, device, NVML_MEMORY_CLK_SAMPLES);

  struct deviceInfo info;
  deviceInfoInitForDevice(&info, device);

  for (;;) {
    samplesBufferFillFromDevice(&pClockSamples, device);
    samplesBufferFillFromDevice(&mClockSamples, device);

    deviceInfoUpdateViolationStatus(&info, device);
    samplesBufferUpdateViolationStatus(&pClockSamples, "pClock", 1);
    samplesBufferUpdateViolationStatus(&mClockSamples, "mClock", 1);

    sleep(1);
  }

  samplesBufferDestroy(&mClockSamples);
  samplesBufferDestroy(&pClockSamples);

  NVML_CHECK(nvmlShutdown());
}
