//
// Copyright © 2025 Agora
// This file is part of TEN Framework, an open source project.
// Licensed under the Apache License, Version 2.0, with certain conditions.
// Refer to the "LICENSE" file in the root directory for more information.
//
#include "include/ten_runtime/common/error_code.h"
#include "include_internal/ten_runtime/binding/nodejs/common/common.h"
#include "include_internal/ten_runtime/binding/nodejs/common/tsfn.h"
#include "include_internal/ten_runtime/binding/nodejs/error/error.h"
#include "include_internal/ten_runtime/binding/nodejs/msg/audio_frame.h"
#include "include_internal/ten_runtime/binding/nodejs/test/env_tester.h"
#include "ten_runtime/test/env_tester.h"
#include "ten_runtime/test/env_tester_proxy.h"
#include "ten_utils/lib/smart_ptr.h"
#include "ten_utils/macro/check.h"
#include "ten_utils/macro/mark.h"
#include "ten_utils/macro/memory.h"

typedef struct ten_env_tester_notify_send_audio_frame_ctx_t {
  ten_shared_ptr_t *c_audio_frame;
  ten_nodejs_tsfn_t *js_cb;
} ten_env_tester_notify_send_audio_frame_ctx_t;

typedef struct ten_nodejs_send_audio_frame_callback_call_ctx_t {
  ten_nodejs_tsfn_t *js_cb;
  ten_error_t *error;
} ten_nodejs_send_audio_frame_callback_call_ctx_t;

static ten_env_tester_notify_send_audio_frame_ctx_t *
ten_env_tester_notify_send_audio_frame_ctx_create(
    ten_shared_ptr_t *c_audio_frame, ten_nodejs_tsfn_t *js_cb) {
  ten_env_tester_notify_send_audio_frame_ctx_t *ctx =
      TEN_MALLOC(sizeof(ten_env_tester_notify_send_audio_frame_ctx_t));
  TEN_ASSERT(ctx, "Failed to allocate memory.");

  ctx->c_audio_frame = c_audio_frame;
  ctx->js_cb = js_cb;

  return ctx;
}

static void ten_env_tester_notify_send_audio_frame_ctx_destroy(
    ten_env_tester_notify_send_audio_frame_ctx_t *ctx) {
  TEN_ASSERT(ctx, "Invalid argument.");

  if (ctx->c_audio_frame) {
    ten_shared_ptr_destroy(ctx->c_audio_frame);
    ctx->c_audio_frame = NULL;
  }

  ctx->js_cb = NULL;

  TEN_FREE(ctx);
}

static ten_nodejs_send_audio_frame_callback_call_ctx_t *
ten_nodejs_send_audio_frame_callback_call_ctx_create(ten_nodejs_tsfn_t *js_cb,
                                                     ten_error_t *error) {
  ten_nodejs_send_audio_frame_callback_call_ctx_t *ctx =
      TEN_MALLOC(sizeof(ten_nodejs_send_audio_frame_callback_call_ctx_t));
  TEN_ASSERT(ctx, "Failed to allocate memory.");

  ctx->js_cb = js_cb;
  ctx->error = error;

  return ctx;
}

static void ten_nodejs_send_audio_frame_callback_call_ctx_destroy(
    ten_nodejs_send_audio_frame_callback_call_ctx_t *ctx) {
  TEN_ASSERT(ctx, "Invalid argument.");

  if (ctx->error) {
    ten_error_destroy(ctx->error);
    ctx->error = NULL;
  }

  TEN_FREE(ctx);
}

static void tsfn_proxy_send_audio_frame_callback(napi_env env, napi_value js_cb,
                                                 TEN_UNUSED void *context,
                                                 void *audio_frame) {
  ten_nodejs_send_audio_frame_callback_call_ctx_t *ctx = audio_frame;
  TEN_ASSERT(ctx, "Should not happen.");

  napi_value js_error = NULL;

  if (ctx->error) {
    js_error = ten_nodejs_error_wrap(env, ctx->error);
    ASSERT_IF_NAPI_FAIL(js_error, "Failed to create JS error", NULL);
  } else {
    js_error = js_undefined(env);
  }

  napi_value argv[] = {js_error};
  napi_status status =
      napi_call_function(env, js_undefined(env), js_cb, 1, argv, NULL);
  ASSERT_IF_NAPI_FAIL(status == napi_ok, "Failed to call JS callback", NULL);

  ten_nodejs_tsfn_release(ctx->js_cb);

  ten_nodejs_send_audio_frame_callback_call_ctx_destroy(ctx);
}

static void proxy_send_audio_frame_callback(
    ten_env_tester_t *ten_env_tester, TEN_UNUSED ten_shared_ptr_t *c_cmd_result,
    void *user_audio_frame, ten_error_t *err) {
  TEN_ASSERT(ten_env_tester, "Should not happen.");
  TEN_ASSERT(ten_env_tester_check_integrity(ten_env_tester, true),
             "Should not happen.");

  ten_env_tester_notify_send_audio_frame_ctx_t *ctx = user_audio_frame;
  TEN_ASSERT(ctx, "Should not happen.");

  ten_nodejs_tsfn_t *js_cb = ctx->js_cb;
  TEN_ASSERT(js_cb && ten_nodejs_tsfn_check_integrity(js_cb, false),
             "Should not happen.");

  ten_error_t *cloned_error = NULL;
  if (err) {
    cloned_error = ten_error_create();
    ten_error_copy(cloned_error, err);
  }

  ten_nodejs_send_audio_frame_callback_call_ctx_t *call_info =
      ten_nodejs_send_audio_frame_callback_call_ctx_create(js_cb, cloned_error);
  TEN_ASSERT(call_info, "Failed to create callback call info.");

  bool rc = ten_nodejs_tsfn_invoke(ctx->js_cb, call_info);
  TEN_ASSERT(rc, "Should not happen.");

  ten_env_tester_notify_send_audio_frame_ctx_destroy(ctx);
}

static void ten_env_tester_proxy_notify_send_audio_frame(
    ten_env_tester_t *ten_env_tester, void *user_audio_frame) {
  TEN_ASSERT(user_audio_frame, "Invalid argument.");
  TEN_ASSERT(ten_env_tester, "Should not happen.");
  TEN_ASSERT(ten_env_tester_check_integrity(ten_env_tester, true),
             "Should not happen.");

  ten_env_tester_notify_send_audio_frame_ctx_t *ctx = user_audio_frame;
  TEN_ASSERT(ctx, "Should not happen.");

  ten_error_t err;
  TEN_ERROR_INIT(err);

  bool rc = ten_env_tester_send_audio_frame(ten_env_tester, ctx->c_audio_frame,
                                            proxy_send_audio_frame_callback,
                                            ctx, &err);
  if (!rc) {
    proxy_send_audio_frame_callback(ten_env_tester, NULL, ctx, &err);
  }

  ten_error_deinit(&err);
}

napi_value ten_nodejs_ten_env_tester_send_audio_frame(napi_env env,
                                                      napi_callback_info info) {
  const size_t argc = 3;
  napi_value args[argc];  // this, audio_frame, callback
  if (!ten_nodejs_get_js_func_args(env, info, args, argc)) {
    napi_fatal_error(NULL, NAPI_AUTO_LENGTH,
                     "Incorrect number of parameters passed.",
                     NAPI_AUTO_LENGTH);
    TEN_ASSERT(0, "Should not happen.");
    return NULL;
  }

  ten_nodejs_ten_env_tester_t *ten_env_tester_bridge = NULL;
  napi_status status =
      napi_unwrap(env, args[0], (void **)&ten_env_tester_bridge);
  RETURN_UNDEFINED_IF_NAPI_FAIL(
      status == napi_ok && ten_env_tester_bridge != NULL,
      "Failed to unwrap TenEnv object");

  TEN_ASSERT(ten_env_tester_bridge, "Should not happen.");
  TEN_ASSERT(
      ten_nodejs_ten_env_tester_check_integrity(ten_env_tester_bridge, true),
      "Should not happen.");

  if (ten_env_tester_bridge->c_ten_env_tester_proxy == NULL) {
    ten_error_t err;
    TEN_ERROR_INIT(err);

    ten_error_set(&err, TEN_ERROR_CODE_TEN_IS_CLOSED,
                  "ten_env_tester.send_audio_frame() failed because "
                  "ten_env_tester is closed.");

    napi_value js_error = ten_nodejs_error_wrap(env, &err);
    RETURN_UNDEFINED_IF_NAPI_FAIL(js_error, "Failed to create JS error");

    ten_error_deinit(&err);

    return js_error;
  }

  ten_nodejs_audio_frame_t *audio_frame_bridge = NULL;
  status = napi_unwrap(env, args[1], (void **)&audio_frame_bridge);
  RETURN_UNDEFINED_IF_NAPI_FAIL(status == napi_ok && audio_frame_bridge != NULL,
                                "Failed to unwrap audio_frame object");

  ten_nodejs_tsfn_t *cb_tsfn = ten_nodejs_tsfn_create(
      env, "[TSFN] TenEnvTester::send_audio_frame callback", args[2],
      tsfn_proxy_send_audio_frame_callback);
  RETURN_UNDEFINED_IF_NAPI_FAIL(cb_tsfn, "Failed to create TSFN");

  ten_error_t err;
  TEN_ERROR_INIT(err);

  ten_env_tester_notify_send_audio_frame_ctx_t *notify_info =
      ten_env_tester_notify_send_audio_frame_ctx_create(
          ten_shared_ptr_clone(audio_frame_bridge->msg.msg), cb_tsfn);
  TEN_ASSERT(notify_info, "Failed to create notify info.");

  bool rc = ten_env_tester_proxy_notify(
      ten_env_tester_bridge->c_ten_env_tester_proxy,
      ten_env_tester_proxy_notify_send_audio_frame, notify_info, &err);
  if (!rc) {
    napi_value js_error = ten_nodejs_error_wrap(env, &err);
    RETURN_UNDEFINED_IF_NAPI_FAIL(js_error, "Failed to create JS error");

    // The JS callback will not be called, so release the TSFN here.
    ten_nodejs_tsfn_release(cb_tsfn);

    ten_env_tester_notify_send_audio_frame_ctx_destroy(notify_info);
    ten_error_deinit(&err);
    return js_error;
  }

  ten_error_deinit(&err);

  return js_undefined(env);
}
